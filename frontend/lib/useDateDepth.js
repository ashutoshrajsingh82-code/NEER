// -----------------------------------------------------------------------------
// NEER Frontend — useDateDepth  (Phase 36A)
//
// The React layer over lib/dateDepthModel.js: owns the reducer, fetches
// GET /dates and GET /model/info through lib/api.js (never fetch() directly),
// and feeds the results into the reducer. All rules — validation, default
// selection, reconciliation, stepping — live in the model; this file only
// does I/O sequencing and exposes a stable, flat API.
//
// Call this ONCE per app (DateDepthProvider does, in the (app) layout) and
// share the result through context — two callers would mean two sets of
// requests and two independent "selected date"s, which is exactly the drift
// this phase exists to remove. Components should use `useDateDepthContext()`.
// -----------------------------------------------------------------------------

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { ApiError, categorizeApiError, dates as fetchDates, modelInfo as fetchModelInfo } from "./api.js";
import {
  canStepDate,
  createInitialState,
  dateDepthActions,
  dateDepthReducer,
  depthOptions,
  isSelectionReady,
  parseDatesPayload,
  parseModelInfoDepths,
  isDateInDataRange,
  selectableDateBounds,
  toIsoDate,
} from "./dateDepthModel.js";

/** Normalizes anything thrown into an ApiError so state never holds a bare Error/undefined. */
function asApiError(caught) {
  return caught instanceof ApiError
    ? caught
    : new ApiError({ code: "unknown_error", message: caught?.message || "An unexpected error occurred." });
}

export function useDateDepth() {
  const [state, dispatch] = useReducer(dateDepthReducer, undefined, createInitialState);
  const [modelInfoData, setModelInfoData] = useState(null);

  // `retry*` bump a counter that the load effects depend on; initial mount
  // is attempt 0. Each attempt's effect cleanup flips `cancelled`, so a slow
  // stale response (or React StrictMode's double-mount) can never land after
  // a newer attempt or after unmount.
  const [datesAttempt, setDatesAttempt] = useState(0);
  const [depthsAttempt, setDepthsAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    fetchDates()
      .then((body) => {
        if (cancelled) return;
        const parsed = parseDatesPayload(body); // api.dates() already validated; defensive
        if (parsed.ok) {
          dispatch(dateDepthActions.datesLoaded(parsed.value));
        } else {
          const error = new ApiError({ code: "invalid_response", message: parsed.reason, status: 200 });
          dispatch(dateDepthActions.datesFailed(error, categorizeApiError(error)));
        }
      })
      .catch((caught) => {
        if (cancelled) return;
        const error = asApiError(caught);
        dispatch(dateDepthActions.datesFailed(error, categorizeApiError(error)));
      });
    return () => {
      cancelled = true;
    };
  }, [datesAttempt]);

  useEffect(() => {
    let cancelled = false;
    fetchModelInfo()
      .then((body) => {
        if (cancelled) return;
        const parsed = parseModelInfoDepths(body);
        if (parsed.ok) {
          setModelInfoData(body);
          dispatch(dateDepthActions.depthsLoaded(parsed.value));
        } else {
          setModelInfoData(null);
          const error = new ApiError({ code: "invalid_response", message: parsed.reason, status: 200 });
          dispatch(dateDepthActions.depthsFailed(error, categorizeApiError(error)));
        }
      })
      .catch((caught) => {
        if (cancelled) return;
        setModelInfoData(null);
        const error = asApiError(caught);
        dispatch(dateDepthActions.depthsFailed(error, categorizeApiError(error)));
      });
    return () => {
      cancelled = true;
    };
  }, [depthsAttempt]);

  const retryDates = useCallback(() => {
    dispatch(dateDepthActions.datesLoading());
    setDatesAttempt((n) => n + 1);
  }, []);
  const retryDepths = useCallback(() => {
    setModelInfoData(null);
    dispatch(dateDepthActions.depthsLoading());
    setDepthsAttempt((n) => n + 1);
  }, []);

  // Setters return whether the request was honoured, judged against the
  // latest state — so callers (and tests) can tell a rejected off-list value
  // from an accepted one without the reducer having to signal it.
  const stateRef = useRef(state);
  stateRef.current = state;

  const selectDate = useCallback((date) => {
    dispatch(dateDepthActions.selectDate(date));
    const iso = toIsoDate(date);
    return iso !== null && isDateInDataRange(iso, stateRef.current.dates.items);
  }, []);
  const selectDepth = useCallback((depth) => {
    dispatch(dateDepthActions.selectDepth(depth));
    return stateRef.current.depths.items.includes(depth);
  }, []);
  const stepDate = useCallback((delta) => dispatch(dateDepthActions.stepDate(delta)), []);

  return useMemo(
    () => ({
      // ---- dates ----
      availableDates: state.dates.items,
      selectableDateBounds: selectableDateBounds(state.dates.items),
      selectedDate: state.selectedDate,
      datesStatus: state.dates.status,
      datesError: state.dates.error,
      datesErrorCategory: state.dates.errorCategory,
      datesMeta: {
        minDate: state.dates.minDate,
        maxDate: state.dates.maxDate,
        dataMode: state.dates.dataMode,
        isSynthetic: state.dates.isSynthetic,
        warnings: state.dates.warnings,
      },
      selectDate,
      stepDate,
      canStepPrev: canStepDate(state, -1),
      canStepNext: canStepDate(state, 1),
      retryDates,

      // ---- depths ----
      availableDepths: state.depths.items,
      depthOptions: depthOptions(state.depths.items),
      selectedDepth: state.selectedDepth,
      depthsStatus: state.depths.status,
      depthsError: state.depths.error,
      depthsErrorCategory: state.depths.errorCategory,
      /** "backend" | "expected_fallback" | null (still loading) */
      depthsSource: state.depths.source,
      depthsValidation: {
        matchesExpected: state.depths.matchesExpected,
        missing: state.depths.missing,
        unexpected: state.depths.unexpected,
        warnings: state.depths.warnings,
      },
      selectDepth,
      retryDepths,

      // ---- combined ----
      /** A valid date AND depth are selected — safe to issue date/depth-parameterised queries. */
      isReady: isSelectionReady(state),
      modelInfoData,
    }),
    [state, selectDate, selectDepth, stepDate, retryDates, retryDepths, modelInfoData]
  );
}

export default useDateDepth;
