"""Native viewer that retains a no-reset environment's first failed state."""

from mjlab.viewer import NativeMujocoViewer


class NoResetMujocoViewer(NativeMujocoViewer):
    """Pause at failure, or close when the caller requested bounded playback.

    A termination must stop the current physics budget immediately: mjlab
    rejects another step until a no-reset environment is explicitly reset.
    """

    def __init__(self, *args, close_on_failure=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_on_failure = close_on_failure
        self.first_failure = None
        self._episode_failed = False
        self._fresh_manual_reset = False

    def _stop_at_failure(self):
        env = self.env.unwrapped
        failed_ids = env.reset_buf.nonzero().flatten().cpu().tolist()
        reasons = {
            str(index): [
                name
                for name in env.termination_manager.active_terms
                if bool(env.termination_manager.get_term(name)[index])
            ]
            for index in failed_ids
        }
        failure = {
            "time_s": self._step_count * env.step_dt,
            "env_ids": failed_ids,
            "reasons": reasons,
        }
        if self.first_failure is None:
            self.first_failure = failure
        self._episode_failed = True
        print(f"Playback stopped at first failure: {failure}. No automatic reset.")
        self.pause()
        if self.close_on_failure:
            self._interrupted = True

    def _execute_step(self):
        if self._episode_failed:
            self.pause()
            return False
        # mjlab creates reset_buf during its first step, not during reset().
        pending = getattr(self.env.unwrapped, "reset_buf", None)
        # Its explicit reset clears the step guard but leaves reset_buf from
        # the preceding episode. The next successful step replaces that buffer.
        fresh_reset = self._fresh_manual_reset
        self._fresh_manual_reset = False
        if not fresh_reset and pending is not None and bool(pending.any()):
            self._stop_at_failure()
            return False
        success = super()._execute_step()
        if not success:
            # The upstream viewer prints the exception and pauses. A bounded
            # invocation must also exit instead of waiting for unreachable steps.
            self._episode_failed = True
            if self.close_on_failure:
                self._interrupted = True
            return False
        if bool(self.env.unwrapped.reset_buf.any()):
            self._stop_at_failure()
            return False
        return True

    def reset_environment(self):
        # This method is only called by the viewer's explicit reset action.
        # Preserve first_failure even if the user starts a new visual episode.
        super().reset_environment()
        self._episode_failed = False
        self._fresh_manual_reset = True
        print(
            "Manual viewer reset: starting a new visual episode; earlier failure remains recorded."
        )
