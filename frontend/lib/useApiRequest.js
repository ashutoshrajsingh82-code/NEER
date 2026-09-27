// NEER Frontend — useApiRequest  (Phase 33D)
//
// The one place a NEER page/component is meant to turn an `frontend/lib/api.js`
// call into on-screen idle/loading/success/error state. `api.js` itself stays
// framework-independent on purpose (see its own header comment) — it has no
// React import and returns plain Promises/throws `ApiError`. This hook is the
// thin React-specific layer on top of that: it owns exactly the state one
// call site needs (via `useState`, scoped to the component that calls the
// hook) and nothing more. There is deliberately no app-wide store, context,
// or singleton here (Requirement 1: "do not create unnecessary global
// state") — two components calling the same endpoint get two independent
// loading states, which is the correct behavior for, say, two KPI cards
// each polling `/health` on their own schedule.
//
// Usage:
//
//   import { dates } from "@/lib/api";
//   import { useApiRequest } from "@/lib/useApiRequest";
//
//   function DateRangeBadge() {
//     const { status, data, error, run } = useApiRequest(dates, { immediate: true });
//     if (status === "loading" || status === "idle") return <Skeleton />;
//     if (status === "error") return <ErrorState message={error.message} onRetry={run} />;
//     return <span>{data.min_date} – {data.max_date}</span>;
//   }
//
// A component that needs parameters supplies them to `run`, not to the hook:
//
//   const { status, data, error, run } = useApiRequest(reconstruct);
//   // later, e.g. from a form submit or a map click:
//   run({ lat, lon, date, depth });
// -----------------------------------------------------------------------------

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "./api";

/** The four states a request can be in. Exported so components can compare
 * against these instead of hand-typing the string literals. */
export const REQUEST_STATUS = Object.freeze({
  IDLE: "idle",
  LOADING: "loading",
  SUCCESS: "success",
  ERROR: "error",
});

/**
 * Wraps any `frontend/lib/api.js` function (or anything with the same
 * "returns a Promise, throws `ApiError`" shape) with idle/loading/success/
 * error state local to the calling component.
 *
 * @template Args, Result
 * @param {(...args: Args) => Promise<Result>} requestFn - an api.js
 *   endpoint function, e.g. `dates`, `reconstruct`, `netcdfExport`
 * @param {object} [config]
 * @param {boolean} [config.immediate=false] - call `requestFn()` (with no
 *   arguments) once, right after mount / whenever `deps` changes. For an
 *   endpoint that takes required parameters, leave this `false` and call
 *   `run(params)` yourself once the parameters are known.
 * @param {ReadonlyArray<*>} [config.deps=[]] - dependency array controlling
 *   when the `immediate` call re-fires (same semantics as `useEffect`'s
 *   second argument). Ignored when `immediate` is `false`.
 * @returns {{
 *   status: "idle"|"loading"|"success"|"error",
 *   data: Result|null,
 *   error: ApiError|null,
 *   isIdle: boolean,
 *   isLoading: boolean,
 *   isSuccess: boolean,
 *   isError: boolean,
 *   run: (...args: Args) => Promise<Result>,
 *   reset: () => void,
 * }}
 */
export function useApiRequest(requestFn, { immediate = false, deps = [] } = {}) {
  const [status, setStatus] = useState(immediate ? REQUEST_STATUS.LOADING : REQUEST_STATUS.IDLE);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  // Guards against a slow, stale request's response landing after a newer
  // one (or after unmount) and clobbering state that no longer applies —
  // e.g. a component that calls `run()` again before the first call
  // resolves, or that unmounts mid-request. Only the most recent call's
  // result is ever applied.
  const latestRequestId = useRef(0);
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const run = useCallback(
    async (...args) => {
      const requestId = ++latestRequestId.current;
      setStatus(REQUEST_STATUS.LOADING);
      setError(null);

      try {
        const result = await requestFn(...args);
        if (mountedRef.current && latestRequestId.current === requestId) {
          setData(result);
          setStatus(REQUEST_STATUS.SUCCESS);
        }
        return result;
      } catch (caught) {
        // requestFn (any api.js export) always throws ApiError, but guard
        // against a non-ApiError escaping (e.g. a programming error in a
        // caller-supplied requestFn) so the UI never gets a `null` .message.
        const normalized =
          caught instanceof ApiError
            ? caught
            : new ApiError({
                code: "unknown_error",
                message: caught?.message || "An unexpected error occurred.",
              });
        if (mountedRef.current && latestRequestId.current === requestId) {
          setError(normalized);
          setStatus(REQUEST_STATUS.ERROR);
        }
        throw normalized;
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [requestFn]
  );

  const reset = useCallback(() => {
    latestRequestId.current += 1; // orphans any in-flight request's result
    setStatus(REQUEST_STATUS.IDLE);
    setData(null);
    setError(null);
  }, []);

  useEffect(() => {
    if (immediate) {
      run().catch(() => {
        // Swallowed here on purpose: the error is already captured in
        // `error`/`status` above for the component to render. Letting it
        // propagate out of this effect would just be an unhandled
        // rejection with nowhere useful to go.
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return {
    status,
    data,
    error,
    isIdle: status === REQUEST_STATUS.IDLE,
    isLoading: status === REQUEST_STATUS.LOADING,
    isSuccess: status === REQUEST_STATUS.SUCCESS,
    isError: status === REQUEST_STATUS.ERROR,
    run,
    reset,
  };
}

export default useApiRequest;