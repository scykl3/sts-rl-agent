"""Vectorized environment: run many :class:`StsEnv` in worker processes.

:class:`SubprocVecEnv` spawns one worker process per engine instance and talks
to them over pipes. Each worker owns a single :class:`StsEnv`; the parent
batches observations, rewards, masks, and done flags across workers so the
learner consumes stacked arrays.

Workers auto-reset on episode end using the pre-1.0 Gymnasium "same-step"
idiom: the observation returned for a done env is the first observation of the
*next* episode, while the terminal observation and info are preserved under
``final_observation`` and ``final_info`` in that env's info dict, and the
``terminated``/``truncated`` flags still refer to the step that ended. This is
not the Gymnasium 1.0 "next-step" reset semantics, and ``infos`` is a per-env
list of dicts, not a batched dict-of-arrays with a ``_final_observation`` mask.
Everything placed in an ``info`` (or observation) must be picklable, since it
crosses the pipe from worker to parent.

The engine is a native extension, so workers use the ``spawn`` start method
(safe with C++ state), and ``make_env`` is shipped to workers via cloudpickle so
closures and lambdas are accepted, not only module-level callables.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from multiprocessing.connection import Connection
from typing import Any

import cloudpickle
import numpy as np

from sts_rl.env.adapter import StsEnv
from sts_rl.interface import InterfaceError, Info, Obs

# A batched observation: same keys as a single Obs, each array gaining a leading
# num_envs axis.
BatchObs = dict[str, np.ndarray]

# Worker command tags.
_CMD_RESET = "reset"
_CMD_STEP = "step"
_CMD_CLOSE = "close"
_CMD_GET_SPACES = "get_spaces"
_CMD_SET_GLOBAL_STEP = "set_global_step"

# Reply status tags.
_OK = "ok"
_ERROR = "error"

_JOIN_TIMEOUT_S = 5.0


class _CloudpickleWrapper:
    """Wrap ``make_env`` so it survives the spawn pickle boundary.

    The default pickler cannot serialize closures or lambdas; cloudpickle can,
    so callers may pass any callable rather than only module-level functions.
    """

    def __init__(self, fn: Callable[[int], StsEnv]) -> None:
        self.fn = fn

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, data: bytes) -> None:
        self.fn = cloudpickle.loads(data)


def _worker(
    remote: Connection, parent_remote: Connection, env_fn: _CloudpickleWrapper, index: int
) -> None:
    """Worker loop: build one env and serve commands until told to close.

    Every reply is ``(status, payload)``; on any exception the worker sends the
    traceback and exits so the parent can re-raise instead of hanging.
    """
    parent_remote.close()
    try:
        env = env_fn.fn(index)
    except Exception:
        import traceback

        remote.send((_ERROR, traceback.format_exc()))
        remote.close()
        return

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == _CMD_STEP:
                obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    final_obs, final_info = obs, info
                    obs, reset_info = env.reset()
                    info = dict(reset_info)
                    info["final_observation"] = final_obs
                    info["final_info"] = final_info
                remote.send((_OK, (obs, reward, terminated, truncated, env.legal_actions(), info)))
            elif cmd == _CMD_RESET:
                obs, info = env.reset(seed=data)
                remote.send((_OK, (obs, env.legal_actions(), info)))
            elif cmd == _CMD_GET_SPACES:
                remote.send((_OK, (env.observation_space, env.action_space)))
            elif cmd == _CMD_SET_GLOBAL_STEP:
                env.set_global_step(data)
                remote.send((_OK, None))
            elif cmd == _CMD_CLOSE:
                env.close()
                remote.send((_OK, None))
                remote.close()
                return
            else:  # pragma: no cover - defensive
                remote.send((_ERROR, f"unknown command {cmd!r}"))
    except Exception:
        import traceback

        remote.send((_ERROR, traceback.format_exc()))
        remote.close()


class SubprocVecEnv:
    """Run ``num_envs`` :class:`StsEnv` in worker processes; batch their I/O.

    Args:
        make_env: factory called once per worker as ``make_env(index)`` to build
            that worker's env. May be a closure or lambda (shipped via
            cloudpickle).
        num_envs: number of parallel worker processes / engine instances.
        start_method: multiprocessing start method; ``spawn`` (the default) is
            required for the native engine.
    """

    def __init__(
        self,
        make_env: Callable[[int], StsEnv],
        num_envs: int,
        *,
        start_method: str = "spawn",
    ) -> None:
        # Set cleanup-relevant attributes first so __del__/close() are safe even
        # if construction fails partway through.
        self.closed = False
        self.remotes: tuple[Connection, ...] = ()
        self.processes: list[mp.process.BaseProcess] = []
        if num_envs < 1:
            raise InterfaceError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = num_envs

        # Typed as Any: typeshed's BaseContext omits Process/Pipe, which the
        # concrete spawn context provides.
        ctx: Any = mp.get_context(start_method)
        pipes = [ctx.Pipe() for _ in range(num_envs)]
        self.remotes = tuple(p[0] for p in pipes)
        work_remotes: tuple[Connection, ...] = tuple(p[1] for p in pipes)
        wrapped = _CloudpickleWrapper(make_env)
        try:
            for index, (work_remote, remote) in enumerate(zip(work_remotes, self.remotes)):
                process = ctx.Process(
                    target=_worker, args=(work_remote, remote, wrapped, index), daemon=True
                )
                process.start()
                # The child owns its end; closing the parent's copy lets the child
                # observe EOF (and thus exit) if the parent dies.
                work_remote.close()
                self.processes.append(process)

            # Query spaces from every worker (not just one) so a make_env that
            # raises in any worker surfaces here at construction, not later.
            self._command_all(_CMD_GET_SPACES, [None] * num_envs)
            spaces = self._recv_all()
            # All workers must expose the same spaces; a heterogeneous make_env is
            # a configuration error and should fail loudly rather than silently
            # batch mismatched shapes.
            if any(worker_spaces != spaces[0] for worker_spaces in spaces[1:]):
                raise InterfaceError("workers reported mismatched observation/action spaces")
        except BaseException:
            # If start() fails partway, the parent still holds the write end of
            # every work_remote it has not closed yet; leaving them open would make
            # close()'s drain block forever (no worker to reply, no EOF). Close
            # them all so close() sees EOF, then tear down.
            for work_remote in work_remotes:
                try:
                    work_remote.close()
                except OSError:
                    pass
            self.close()
            raise
        self.observation_space, self.action_space = spaces[0]

    # -- Vector API ---------------------------------------------------------

    def reset(self, seeds: Sequence[int] | None = None) -> tuple[BatchObs, np.ndarray]:
        """Reset every env; return stacked obs and a ``(num_envs, ACTION_DIM)`` mask."""
        self._assert_open()
        seed_list = self._normalize_seeds(seeds)
        self._command_all(_CMD_RESET, seed_list)
        obs_list, masks, _infos = zip(*self._recv_all())
        return self._stack_obs(obs_list), np.stack(masks)

    def step(
        self, actions: Sequence[int] | np.ndarray
    ) -> tuple[BatchObs, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Info]]:
        """Step every env (auto-resetting any that finish); return stacked results.

        Returns ``(obs, rewards, terminated, truncated, masks, infos)`` where
        ``masks`` is ``(num_envs, ACTION_DIM)``, symmetric with :meth:`reset`.
        """
        self._assert_open()
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,):
            raise InterfaceError(f"actions must have shape ({self.num_envs},), got {actions.shape}")
        self._command_all(_CMD_STEP, [int(a) for a in actions])
        obs_list, rewards, terminated, truncated, masks, infos = zip(*self._recv_all())
        return (
            self._stack_obs(obs_list),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=np.bool_),
            np.asarray(truncated, dtype=np.bool_),
            np.stack(masks),
            list(infos),
        )

    def set_global_step(self, t: int) -> None:
        """Broadcast the shared global env-step count to every worker.

        Keeps the shaping anneal ``beta(t)`` on its intended horizon under
        parallel workers, where each worker's own step count understates total
        interactions (see :meth:`StsEnv.set_global_step`).
        """
        self._assert_open()
        self._command_all(_CMD_SET_GLOBAL_STEP, [t] * self.num_envs)
        self._recv_all()

    def close(self) -> None:
        """Terminate all workers and release their engines. Idempotent."""
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send((_CMD_CLOSE, None))
            except (BrokenPipeError, EOFError, OSError):
                pass  # worker already gone
        for index, remote in enumerate(self.remotes):
            try:
                # poll() before recv() so a worker that never started (write end
                # already closed -> EOF) or one that is wedged (poll times out)
                # can never block the drain; only reply or EOF/error get here.
                if remote.poll(_JOIN_TIMEOUT_S):
                    self._recv(remote, index)
            except (EOFError, OSError, RuntimeError):
                pass
            try:
                remote.close()
            except OSError:
                pass
        for process in self.processes:
            process.join(timeout=_JOIN_TIMEOUT_S)
            if process.is_alive():  # pragma: no cover - only on a hung worker
                process.terminate()
                process.join(timeout=_JOIN_TIMEOUT_S)
            if process.is_alive():  # pragma: no cover - worker ignored SIGTERM
                process.kill()
                process.join()
        self.closed = True

    # -- Context manager / cleanup -----------------------------------------

    def __enter__(self) -> SubprocVecEnv:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort: never raise from the finalizer.
        try:
            self.close()
        except Exception:  # pragma: no cover - finalizer must not raise
            pass

    # -- Internals ----------------------------------------------------------

    def _assert_open(self) -> None:
        if self.closed:
            raise InterfaceError("operation on a closed SubprocVecEnv")

    def _normalize_seeds(self, seeds: Sequence[int] | None) -> list[int | None]:
        if seeds is None:
            return [None] * self.num_envs
        seed_list: list[int | None] = list(seeds)
        if len(seed_list) != self.num_envs:
            raise InterfaceError(f"expected {self.num_envs} seeds, got {len(seed_list)}")
        return seed_list

    def _command_all(self, cmd: str, datas: Sequence[Any]) -> None:
        """Send one command to every worker; tear down and raise if a send fails.

        A failed send means a worker has already died. Rather than leaving the
        envs desynced (some advanced, some not), close the whole vec env and
        raise so the caller cannot issue a mismatched follow-up.
        """
        for index, (remote, data) in enumerate(zip(self.remotes, datas)):
            try:
                remote.send((cmd, data))
            except (BrokenPipeError, EOFError, OSError) as exc:
                self.close()
                raise RuntimeError(
                    f"SubprocVecEnv worker {index} is not accepting commands"
                ) from exc

    def _recv_all(self) -> list[Any]:
        return [self._recv(remote, index) for index, remote in enumerate(self.remotes)]

    @staticmethod
    def _stack_obs(obs_list: Sequence[Obs]) -> BatchObs:
        return {key: np.stack([obs[key] for obs in obs_list]) for key in obs_list[0]}

    @staticmethod
    def _recv(remote: Connection, index: int) -> Any:
        try:
            status, payload = remote.recv()
        except EOFError as exc:
            # Peer closed without replying: the worker crashed (segfault,
            # os._exit, ...). Surface it like the caught-exception path instead
            # of blocking or leaking a bare EOFError.
            raise RuntimeError(
                f"SubprocVecEnv worker {index} died without sending a reply"
            ) from exc
        if status == _ERROR:
            raise RuntimeError(f"SubprocVecEnv worker {index} failed:\n{payload}")
        return payload
