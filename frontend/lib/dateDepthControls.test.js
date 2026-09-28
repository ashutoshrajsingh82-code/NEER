import { describe, expect, it } from "vitest";
import {
  addDays,
  addMonths,
  buildMonthGrid,
  canShiftMonth,
  clampMonth,
  countAvailableInMonth,
  depthToIndex,
  describeDepth,
  formatDepthMetres,
  indexToDepth,
  initialFocusDate,
  initialMonth,
  monthRange,
  moveCalendarFocus,
  navigateDepth,
  weekdayIndex,
  yearsWithDates,
} from "./dateDepthControls";
import { DEPTH_LEVELS } from "./oceanDomain";
import {
  canStepDate,
  createInitialState,
  dateDepthActions,
  dateDepthReducer,
  parseDatesPayload,
  parseModelInfoDepths,
} from "./dateDepthModel";

const DATES = ["2020-01-01", "2020-01-02", "2020-01-05", "2020-02-10", "2020-04-01"];
const set = new Set(DATES);

describe("ISO date arithmetic", () => {
  it("adds days across month/year boundaries in UTC", () => {
    expect(addDays("2020-01-31", 1)).toBe("2020-02-01");
    expect(addDays("2020-03-01", -1)).toBe("2020-02-29");
    expect(addDays("2020-12-31", 1)).toBe("2021-01-01");
  });
  it("adds months clamping the day", () => {
    expect(addMonths("2020-01-31", 1)).toBe("2020-02-29");
    expect(addMonths("2021-01-31", 1)).toBe("2021-02-28");
    expect(addMonths("2020-01-15", -1)).toBe("2019-12-15");
  });
  it("numbers weekdays Monday-first", () => {
    expect(weekdayIndex("2020-01-01")).toBe(2); // Wednesday
    expect(weekdayIndex("2020-01-05")).toBe(6); // Sunday
  });
});

describe("month grid", () => {
  const weeks = buildMonthGrid({ year: 2020, month: 1 }, set, "2020-01-02");
  it("has whole weeks of 7 and pads with nulls", () => {
    expect(weeks.every((w) => w.length === 7)).toBe(true);
    expect(weeks[0].slice(0, 2)).toEqual([null, null]); // Jan 1 2020 is a Wednesday
    expect(weeks[0][2].iso).toBe("2020-01-01");
  });
  it("marks only backend-listed dates available and flags the selection", () => {
    const cells = weeks.flat().filter(Boolean);
    expect(cells).toHaveLength(31);
    expect(cells.filter((c) => c.available).map((c) => c.iso)).toEqual(["2020-01-01", "2020-01-02", "2020-01-05"]);
    expect(cells.filter((c) => c.selected).map((c) => c.iso)).toEqual(["2020-01-02"]);
  });
  it("counts available dates per month", () => {
    expect(countAvailableInMonth({ year: 2020, month: 1 }, set)).toBe(3);
    expect(countAvailableInMonth({ year: 2020, month: 3 }, set)).toBe(0);
  });
});

describe("month range", () => {
  const range = monthRange(DATES);
  it("spans first to last month with data", () => {
    expect(range).toEqual({ first: { year: 2020, month: 1 }, last: { year: 2020, month: 4 } });
    expect(monthRange([])).toBeNull();
  });
  it("clamps and gates month navigation", () => {
    expect(clampMonth({ year: 2019, month: 6 }, range)).toEqual(range.first);
    expect(clampMonth({ year: 2021, month: 1 }, range)).toEqual(range.last);
    expect(canShiftMonth(range.first, -1, range)).toBe(false);
    expect(canShiftMonth(range.first, 1, range)).toBe(true);
    expect(canShiftMonth(range.last, 1, range)).toBe(false);
  });
  it("lists years with data and picks the opening month", () => {
    expect(yearsWithDates(["2019-05-01", "2020-01-01", "2020-02-01"])).toEqual([2019, 2020]);
    expect(initialMonth("2020-02-10", DATES)).toEqual({ year: 2020, month: 2 });
    expect(initialMonth(null, DATES)).toEqual({ year: 2020, month: 4 });
    expect(initialMonth(null, [])).toBeNull();
  });
});

describe("calendar keyboard navigation", () => {
  const range = monthRange(DATES);
  it("moves by day and week", () => {
    expect(moveCalendarFocus("2020-01-10", "ArrowRight", range)).toBe("2020-01-11");
    expect(moveCalendarFocus("2020-01-10", "ArrowUp", range)).toBe("2020-01-03");
    expect(moveCalendarFocus("2020-01-10", "ArrowDown", range)).toBe("2020-01-17");
  });
  it("Home/End go to the start/end of the week", () => {
    expect(moveCalendarFocus("2020-01-10", "Home", range)).toBe("2020-01-06");
    expect(moveCalendarFocus("2020-01-10", "End", range)).toBe("2020-01-12");
  });
  it("PageUp/PageDown move a month, Shift a year", () => {
    expect(moveCalendarFocus("2020-02-10", "PageDown", range)).toBe("2020-03-10");
    expect(moveCalendarFocus("2020-02-10", "PageUp", range)).toBe("2020-01-10");
  });
  it("never leaves the navigable range", () => {
    expect(moveCalendarFocus("2020-01-01", "ArrowLeft", range)).toBe("2020-01-01");
    expect(moveCalendarFocus("2020-04-30", "ArrowRight", range)).toBe("2020-04-30");
    expect(moveCalendarFocus("2020-02-10", "PageDown", range, true)).toBe("2020-02-10");
  });
  it("ignores other keys", () => {
    expect(moveCalendarFocus("2020-01-10", "a", range)).toBe("2020-01-10");
  });
  it("opens focused on the selection, else the first available day of the month", () => {
    expect(initialFocusDate("2020-01-02", { year: 2020, month: 1 }, set)).toBe("2020-01-02");
    expect(initialFocusDate("2020-01-02", { year: 2020, month: 2 }, set)).toBe("2020-02-10");
    expect(initialFocusDate(null, { year: 2020, month: 3 }, set)).toBe("2020-03-01");
  });
});

describe("depth slider — index positions over the real, uneven level list", () => {
  const depths = [...DEPTH_LEVELS];
  it("the supported list is exactly the 15 NEER levels", () => {
    expect(depths).toEqual([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]);
  });
  it("round-trips every level through its slider index", () => {
    depths.forEach((d, i) => {
      expect(depthToIndex(depths, d)).toBe(i);
      expect(indexToDepth(depths, i)).toBe(d);
    });
  });
  it("can only ever resolve to a listed depth", () => {
    expect(indexToDepth(depths, 3.4)).toBe(20);
    expect(indexToDepth(depths, -5)).toBe(0);
    expect(indexToDepth(depths, 99)).toBe(1000);
    expect(indexToDepth(depths, NaN)).toBeNull();
    expect(indexToDepth([], 0)).toBeNull();
    expect(depthToIndex(depths, 42)).toBe(-1);
    expect(depthToIndex(depths, null)).toBe(-1);
  });
  it("works for a backend list that differs from the expected one", () => {
    const custom = [0, 10, 500];
    expect(indexToDepth(custom, 2)).toBe(500);
    expect(navigateDepth(custom, 10, "ArrowRight")).toBe(500);
  });
});

describe("depth chip keyboard model", () => {
  const depths = [...DEPTH_LEVELS];
  it("arrows step one level and stop at the ends", () => {
    expect(navigateDepth(depths, 30, "ArrowRight")).toBe(50);
    expect(navigateDepth(depths, 30, "ArrowLeft")).toBe(20);
    expect(navigateDepth(depths, 1000, "ArrowRight")).toBe(1000);
    expect(navigateDepth(depths, 0, "ArrowLeft")).toBe(0);
  });
  it("Home/End jump to the extremes; other keys change nothing", () => {
    expect(navigateDepth(depths, 100, "Home")).toBe(0);
    expect(navigateDepth(depths, 100, "End")).toBe(1000);
    expect(navigateDepth(depths, 100, "x")).toBe(100);
    expect(navigateDepth([], 0, "Home")).toBeNull();
  });
  it("formats with scientific units", () => {
    expect(formatDepthMetres(0)).toBe("0 m");
    expect(formatDepthMetres(1000)).toBe("1,000 m");
    expect(formatDepthMetres(null)).toBe("--");
    expect(describeDepth(0)).toBe("Surface, 0 metres");
    expect(describeDepth(75)).toBe("75 metres");
  });
});

// The controls only ever dispatch these actions; this proves the state they
// drive stays consistent (single source of truth) for the scenarios the phase
// spec lists.
describe("controls -> shared state", () => {
  function loaded() {
    let s = createInitialState();
    s = dateDepthReducer(s, dateDepthActions.datesLoaded(parseDatesPayload({ dates: DATES }).value));
    s = dateDepthReducer(
      s,
      dateDepthActions.depthsLoaded(parseModelInfoDepths({ architecture: { depths: [...DEPTH_LEVELS] } }).value)
    );
    return s;
  }
  it("opens on the latest date and the surface", () => {
    const s = loaded();
    expect(s.selectedDate).toBe("2020-04-01");
    expect(s.selectedDepth).toBe(0);
  });
  it("previous/next skip gaps and stop at the first/last available date", () => {
    let s = loaded();
    expect(canStepDate(s, 1)).toBe(false);
    s = dateDepthReducer(s, dateDepthActions.stepDate(-1));
    expect(s.selectedDate).toBe("2020-02-10"); // not 2020-03-31
    s = dateDepthReducer(s, dateDepthActions.selectDate("2020-01-01"));
    expect(canStepDate(s, -1)).toBe(false);
    expect(dateDepthReducer(s, dateDepthActions.stepDate(-1))).toBe(s);
    expect(dateDepthReducer(s, dateDepthActions.stepDate(1)).selectedDate).toBe("2020-01-02");
  });
  it("an unavailable date can never be selected", () => {
    const s = loaded();
    expect(dateDepthReducer(s, dateDepthActions.selectDate("2020-03-15"))).toBe(s);
  });
  it("slider and chips both resolve to the same list-backed depth", () => {
    let s = loaded();
    s = dateDepthReducer(s, dateDepthActions.selectDepth(indexToDepth(s.depths.items, 9)));
    expect(s.selectedDepth).toBe(150);
    s = dateDepthReducer(s, dateDepthActions.selectDepth(navigateDepth(s.depths.items, s.selectedDepth, "End")));
    expect(s.selectedDepth).toBe(1000);
    expect(dateDepthReducer(s, dateDepthActions.selectDepth(42))).toBe(s);
  });
});