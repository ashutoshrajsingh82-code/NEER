// NEER Frontend — date/depth model test suite  (Phase 36A)
//
// Pure-function tests for lib/dateDepthModel.js: payload validation, depth
// validation against the expected NEER configuration, and the reducer's
// selection/reconciliation rules. No React, no network.
import { describe, expect, it } from "vitest";
import {
  EXPECTED_DEPTHS,
  LOAD_STATUS,
  DEPTH_SOURCE,
  canStepDate,
  compareWithExpectedDepths,
  createInitialState,
  dateDepthActions as A,
  dateDepthReducer,
  depthOptions,
  isIsoDate,
  isSelectionReady,
  parseDatesPayload,
  parseModelInfoDepths,
  reconcileDate,
  reconcileDepth,
  toIsoDate,
  validateDepthList,
} from "./dateDepthModel.js";

const EXPECTED = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];
const datesBody = (dates, extra = {}) => ({
  dates,
  count: dates.length,
  min_date: dates[0] ?? null,
  max_date: dates[dates.length - 1] ?? null,
  data_mode: "REAL",
  is_synthetic: false,
  ...extra,
});
const modelInfoBody = (depths, extra = {}) => ({
  architecture: { depths, num_depths: depths.length, in_channels: 8 },
  runtime: {},
  checkpoint: {},
  ...extra,
});
const reduce = (...actions) => actions.reduce(dateDepthReducer, createInitialState());
const loadedDates = (list) => A.datesLoaded(parseDatesPayload(datesBody(list)).value);
const loadedDepths = (list) => A.depthsLoaded(parseModelInfoDepths(modelInfoBody(list)).value);

describe("expected depth configuration", () => {
  it("is exactly the 15 NEER model depths from the spec", () => {
    expect([...EXPECTED_DEPTHS]).toEqual(EXPECTED);
  });
});

describe("isIsoDate / toIsoDate", () => {
  it("accepts real calendar dates only", () => {
    expect(isIsoDate("2020-02-29")).toBe(true);
    expect(isIsoDate("2021-02-29")).toBe(false); // not a leap year
    expect(isIsoDate("2020-13-01")).toBe(false);
    expect(isIsoDate("2020-1-1")).toBe(false);
    expect(isIsoDate("2020-01-01T00:00:00Z")).toBe(false);
    expect(isIsoDate(20200101)).toBe(false);
    expect(isIsoDate(null)).toBe(false);
  });
  it("normalizes Date instances in UTC and rejects invalid ones", () => {
    expect(toIsoDate(new Date("2020-01-15T00:00:00Z"))).toBe("2020-01-15");
    expect(toIsoDate(new Date("nope"))).toBeNull();
    expect(toIsoDate("2020-01-15")).toBe("2020-01-15");
    expect(toIsoDate("15/01/2020")).toBeNull();
  });
});

describe("parseDatesPayload", () => {
  it("returns a sorted, de-duplicated list plus derived min/max/count", () => {
    const r = parseDatesPayload(datesBody(["2020-01-03", "2020-01-01", "2020-01-02", "2020-01-02"], { count: 4 }));
    expect(r.ok).toBe(true);
    expect(r.value.dates).toEqual(["2020-01-01", "2020-01-02", "2020-01-03"]);
    expect(r.value).toMatchObject({ count: 3, minDate: "2020-01-01", maxDate: "2020-01-03", dataMode: "REAL", isSynthetic: false });
    expect(r.value.warnings.join(" ")).toMatch(/duplicate/);
    expect(r.value.warnings.join(" ")).toMatch(/count=4/);
  });
  it("treats an empty list as VALID (the empty state, not an error)", () => {
    const r = parseDatesPayload(datesBody([]));
    expect(r.ok).toBe(true);
    expect(r.value).toMatchObject({ dates: [], count: 0, minDate: null, maxDate: null });
  });
  it("warns (does not fail) when min/max disagree with the list", () => {
    const r = parseDatesPayload(datesBody(["2020-01-01", "2020-01-02"], { max_date: "2021-01-01" }));
    expect(r.ok).toBe(true);
    expect(r.value.maxDate).toBe("2020-01-02"); // the list wins
    expect(r.value.warnings.join(" ")).toMatch(/max_date/);
  });
  it.each([
    ["null", null],
    ["array", []],
    ["string", "dates"],
    ["missing dates", { count: 0 }],
    ["dates not an array", { dates: "2020-01-01" }],
    ["non-string entry", { dates: ["2020-01-01", 20200102] }],
    ["malformed entry", { dates: ["2020-01-01", "yesterday"] }],
    ["impossible date", { dates: ["2020-02-30"] }],
  ])("rejects an invalid payload: %s", (_label, body) => {
    const r = parseDatesPayload(body);
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/dates/i);
  });
  it("defaults dataMode/isSynthetic when the backend omits them", () => {
    const r = parseDatesPayload({ dates: ["2020-01-01"] });
    expect(r.value).toMatchObject({ dataMode: "UNKNOWN", isSynthetic: false });
  });
});

describe("validateDepthList / compareWithExpectedDepths", () => {
  it("accepts the expected list", () => {
    expect(validateDepthList(EXPECTED)).toEqual({ ok: true, depths: EXPECTED });
  });
  it.each([
    ["undefined", undefined],
    ["empty", []],
    ["string entry", [0, "5"]],
    ["NaN", [0, NaN]],
    ["negative", [-5, 0]],
    ["duplicate", [0, 5, 5]],
    ["unsorted", [0, 10, 5]],
  ])("rejects: %s", (_l, list) => {
    expect(validateDepthList(list).ok).toBe(false);
  });
  it("reports missing/unexpected levels against the expected configuration", () => {
    expect(compareWithExpectedDepths(EXPECTED)).toEqual({ matches: true, missing: [], unexpected: [] });
    const cmp = compareWithExpectedDepths([0, 5, 10, 15, 1000]);
    expect(cmp.matches).toBe(false);
    expect(cmp.unexpected).toEqual([15]);
    expect(cmp.missing).toContain(20);
    expect(cmp.missing).not.toContain(0);
  });
});

describe("parseModelInfoDepths", () => {
  it("uses the backend list when it matches the expected configuration", () => {
    const r = parseModelInfoDepths(modelInfoBody(EXPECTED));
    expect(r.ok).toBe(true);
    expect(r.value).toMatchObject({ depths: EXPECTED, matchesExpected: true, warnings: [] });
  });
  it("uses a valid-but-different backend list as-is and flags the mismatch (never silently replaced)", () => {
    const r = parseModelInfoDepths(modelInfoBody([0, 10, 100]));
    expect(r.ok).toBe(true);
    expect(r.value.depths).toEqual([0, 10, 100]);
    expect(r.value.matchesExpected).toBe(false);
    expect(r.value.missing).toContain(5);
    expect(r.value.warnings[0]).toMatch(/differs from the expected/);
  });
  it("rejects num_depths that contradicts the list", () => {
    const body = modelInfoBody([0, 5, 10]);
    body.architecture.num_depths = 15;
    expect(parseModelInfoDepths(body).ok).toBe(false);
  });
  it.each([
    ["null", null],
    ["no architecture", { runtime: {} }],
    ["architecture not object", { architecture: "x" }],
    ["no depths", { architecture: { num_depths: 15 } }],
    ["bad depths", { architecture: { depths: [0, "x"] } }],
  ])("rejects: %s", (_l, body) => {
    expect(parseModelInfoDepths(body).ok).toBe(false);
  });
});

describe("reconcileDate / reconcileDepth", () => {
  it("defaults to the latest date, keeps a still-valid current date, and handles empty", () => {
    const list = ["2020-01-01", "2020-01-02", "2020-01-03"];
    expect(reconcileDate(list, null)).toBe("2020-01-03");
    expect(reconcileDate(list, "2020-01-01")).toBe("2020-01-01");
    expect(reconcileDate(list, "1999-01-01")).toBe("2020-01-03");
    expect(reconcileDate([], "2020-01-01")).toBeNull();
  });
  it("defaults to the surface, keeps a still-valid depth, falls back to shallowest, handles empty", () => {
    expect(reconcileDepth(EXPECTED, null)).toBe(0);
    expect(reconcileDepth(EXPECTED, 200)).toBe(200);
    expect(reconcileDepth(EXPECTED, 42)).toBe(0);
    expect(reconcileDepth([10, 20], null)).toBe(10);
    expect(reconcileDepth([], 0)).toBeNull();
  });
});

describe("dateDepthReducer — dates", () => {
  it("starts loading with nothing selected", () => {
    const s = createInitialState();
    expect(s.dates.status).toBe(LOAD_STATUS.LOADING);
    expect(s.depths.status).toBe(LOAD_STATUS.LOADING);
    expect(s.selectedDate).toBeNull();
    expect(s.selectedDepth).toBeNull();
    expect(isSelectionReady(s)).toBe(false);
  });
  it("success: stores dates and selects the latest", () => {
    const s = reduce(loadedDates(["2020-01-01", "2020-01-02"]));
    expect(s.dates.status).toBe(LOAD_STATUS.SUCCESS);
    expect(s.dates.items).toEqual(["2020-01-01", "2020-01-02"]);
    expect(s.selectedDate).toBe("2020-01-02");
  });
  it("empty: status empty, nothing selected", () => {
    const s = reduce(loadedDates([]));
    expect(s.dates.status).toBe(LOAD_STATUS.EMPTY);
    expect(s.selectedDate).toBeNull();
  });
  it("error: clears dates AND selection, keeps the error + category", () => {
    const err = new Error("boom");
    const s = reduce(loadedDates(["2020-01-01"]), A.datesFailed(err, "network"));
    expect(s.dates).toMatchObject({ status: LOAD_STATUS.ERROR, items: [], error: err, errorCategory: "network" });
    expect(s.selectedDate).toBeNull();
  });
  it("retry: loading clears the error but a later success restores state", () => {
    const s = reduce(A.datesFailed(new Error("x"), "network"), A.datesLoading(), loadedDates(["2020-05-05"]));
    expect(s.dates.status).toBe(LOAD_STATUS.SUCCESS);
    expect(s.dates.error).toBeNull();
    expect(s.selectedDate).toBe("2020-05-05");
  });
  it("a reload keeps the user's selection if it's still available, else resets to latest", () => {
    const base = reduce(loadedDates(["2020-01-01", "2020-01-02", "2020-01-03"]), A.selectDate("2020-01-01"));
    expect(dateDepthReducer(base, loadedDates(["2020-01-01", "2020-01-05"])).selectedDate).toBe("2020-01-01");
    expect(dateDepthReducer(base, loadedDates(["2021-01-01", "2021-01-02"])).selectedDate).toBe("2021-01-02");
  });
  it("selectDate accepts only listed dates (string or Date), ignores everything else", () => {
    const s0 = reduce(loadedDates(["2020-01-01", "2020-01-02", "2020-01-03"]));
    expect(dateDepthReducer(s0, A.selectDate("2020-01-01")).selectedDate).toBe("2020-01-01");
    expect(dateDepthReducer(s0, A.selectDate(new Date("2020-01-02T00:00:00Z"))).selectedDate).toBe("2020-01-02");
    expect(dateDepthReducer(s0, A.selectDate("2020-06-06"))).toBe(s0); // not listed
    expect(dateDepthReducer(s0, A.selectDate("garbage"))).toBe(s0);
    expect(dateDepthReducer(s0, A.selectDate(null))).toBe(s0);
  });
  it("stepDate moves across AVAILABLE dates (not calendar days) and stops at the ends", () => {
    let s = reduce(loadedDates(["2020-01-01", "2020-01-15", "2020-03-01"]));
    expect(s.selectedDate).toBe("2020-03-01");
    expect(canStepDate(s, 1)).toBe(false);
    expect(canStepDate(s, -1)).toBe(true);
    expect(dateDepthReducer(s, A.stepDate(1))).toBe(s);
    s = dateDepthReducer(s, A.stepDate(-1));
    expect(s.selectedDate).toBe("2020-01-15"); // skipped straight over the gap
    s = dateDepthReducer(s, A.stepDate(-1));
    expect(s.selectedDate).toBe("2020-01-01");
    expect(canStepDate(s, -1)).toBe(false);
    expect(dateDepthReducer(s, A.stepDate(-1))).toBe(s);
  });
  it("stepDate/canStepDate are inert with no selection", () => {
    const s = createInitialState();
    expect(dateDepthReducer(s, A.stepDate(1))).toBe(s);
    expect(canStepDate(s, 1)).toBe(false);
  });
});

describe("dateDepthReducer — depths", () => {
  it("success: uses the backend list, selects the surface, source=backend", () => {
    const s = reduce(loadedDepths(EXPECTED));
    expect(s.depths).toMatchObject({ status: LOAD_STATUS.SUCCESS, items: EXPECTED, source: DEPTH_SOURCE.BACKEND, matchesExpected: true });
    expect(s.selectedDepth).toBe(0);
  });
  it("a mismatching backend list is used as-is and flagged", () => {
    const s = reduce(loadedDepths([0, 10, 100]));
    expect(s.depths.items).toEqual([0, 10, 100]);
    expect(s.depths.matchesExpected).toBe(false);
    expect(s.depths.source).toBe(DEPTH_SOURCE.BACKEND);
  });
  it("failure: falls back to the EXPECTED list, explicitly labelled, error retained", () => {
    const err = new Error("503");
    const s = reduce(A.depthsFailed(err, "api"));
    expect(s.depths).toMatchObject({
      status: LOAD_STATUS.ERROR,
      items: EXPECTED,
      source: DEPTH_SOURCE.EXPECTED_FALLBACK,
      error: err,
      errorCategory: "api",
    });
    expect(s.selectedDepth).toBe(0);
  });
  it("recovering from fallback to the backend list keeps a still-valid selection", () => {
    const s = reduce(A.depthsFailed(new Error("x"), "network"), A.selectDepth(200), A.depthsLoading(), loadedDepths(EXPECTED));
    expect(s.depths.source).toBe(DEPTH_SOURCE.BACKEND);
    expect(s.selectedDepth).toBe(200);
  });
  it("a new list that drops the selected depth reconciles to the surface", () => {
    const s = reduce(loadedDepths(EXPECTED), A.selectDepth(1000), loadedDepths([0, 5, 10]));
    expect(s.selectedDepth).toBe(0);
  });
  it("selectDepth accepts only listed depths (no snapping, no strings)", () => {
    const s0 = reduce(loadedDepths(EXPECTED));
    expect(dateDepthReducer(s0, A.selectDepth(300)).selectedDepth).toBe(300);
    expect(dateDepthReducer(s0, A.selectDepth(42))).toBe(s0);
    expect(dateDepthReducer(s0, A.selectDepth("300"))).toBe(s0);
    expect(dateDepthReducer(s0, A.selectDepth(null))).toBe(s0);
  });
});

describe("selection readiness + options", () => {
  it("is ready only when both a date and a depth are selected", () => {
    expect(isSelectionReady(reduce(loadedDates(["2020-01-01"])))).toBe(false);
    expect(isSelectionReady(reduce(loadedDepths(EXPECTED)))).toBe(false);
    expect(isSelectionReady(reduce(loadedDates(["2020-01-01"]), loadedDepths(EXPECTED)))).toBe(true);
  });
  it("is independent of arrival order", () => {
    const a = reduce(loadedDates(["2020-01-01"]), loadedDepths(EXPECTED));
    const b = reduce(loadedDepths(EXPECTED), loadedDates(["2020-01-01"]));
    expect(a).toEqual(b);
  });
  it("dates failing does not disturb the depth state, and vice versa", () => {
    const s = reduce(loadedDepths(EXPECTED), A.datesFailed(new Error("x"), "network"));
    expect(s.depths.status).toBe(LOAD_STATUS.SUCCESS);
    expect(s.selectedDepth).toBe(0);
    expect(isSelectionReady(s)).toBe(false);
  });
  it("depthOptions labels 0 as Surface and the rest in metres", () => {
    expect(depthOptions([0, 5, 1000])).toEqual([
      { value: 0, label: "Surface" },
      { value: 5, label: "5 m" },
      { value: 1000, label: "1000 m" },
    ]);
  });
});