// -----------------------------------------------------------------------------
// NEER Frontend — date/depth CONTROL logic  (Phase 36B)
//
// Pure helpers behind the dashboard's calendar, previous/next buttons, depth
// slider and depth chips. State itself (available/selected date and depth and
// the rules for changing them) stays in lib/dateDepthModel.js — this file only
// answers the questions a *control* needs answered:
//
//   - which cells does the month view show, and which of them are selectable?
//   - where does keyboard focus go when an arrow key is pressed?
//   - what month range may the calendar navigate within?
//   - what slider position does a depth correspond to, and back?
//   - which depth does an arrow/Home/End key on the chips select?
//
// No React, no DOM, no fetch — runs under the repo's node-only vitest setup.
//
// Dates are always "YYYY-MM-DD" strings and all arithmetic is done in UTC, the
// same convention as dateDepthModel.js / api.js, so a calendar cell can never
// drift by a day in a viewer's local time zone.
// -----------------------------------------------------------------------------

import { isIsoDate } from "./dateDepthModel.js";

const pad = (n) => String(n).padStart(2, "0");

export const WEEKDAY_LABELS = Object.freeze(["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]);
export const MONTH_NAMES = Object.freeze([
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
]);

// ---------------------------------------------------------------------------
// ISO date arithmetic (UTC)
// ---------------------------------------------------------------------------

/** @param {string} iso @returns {{year:number, month:number, day:number}} month is 1-12 */
export function parseIso(iso) {
  const [year, month, day] = iso.split("-").map(Number);
  return { year, month, day };
}

export function formatIso(year, month, day) {
  return `${String(year).padStart(4, "0")}-${pad(month)}-${pad(day)}`;
}

export function daysInMonth(year, month) {
  return new Date(Date.UTC(year, month, 0)).getUTCDate();
}

/** @param {string} iso @param {number} n @returns {string} */
export function addDays(iso, n) {
  const { year, month, day } = parseIso(iso);
  return new Date(Date.UTC(year, month - 1, day + n)).toISOString().slice(0, 10);
}

/** Adds months, clamping the day (31 Jan + 1 month = 28/29 Feb). */
export function addMonths(iso, n) {
  const { year, month, day } = parseIso(iso);
  const index = year * 12 + (month - 1) + n;
  const y = Math.floor(index / 12);
  const m = (index % 12) + 1;
  return formatIso(y, m, Math.min(day, daysInMonth(y, m)));
}

/** Monday = 0 … Sunday = 6. */
export function weekdayIndex(iso) {
  const { year, month, day } = parseIso(iso);
  return (new Date(Date.UTC(year, month - 1, day)).getUTCDay() + 6) % 7;
}

// ---------------------------------------------------------------------------
// Month navigation
// ---------------------------------------------------------------------------

/** @typedef {{year:number, month:number}} YearMonth  month is 1-12 */

/** @param {string} iso @returns {YearMonth} */
export function monthOf(iso) {
  const { year, month } = parseIso(iso);
  return { year, month };
}

/** @param {YearMonth} a @param {YearMonth} b @returns {number} <0, 0, >0 */
export function compareMonths(a, b) {
  return a.year * 12 + a.month - (b.year * 12 + b.month);
}

/** @param {YearMonth} ym @param {number} delta @returns {YearMonth} */
export function shiftMonth(ym, delta) {
  const index = ym.year * 12 + (ym.month - 1) + delta;
  return { year: Math.floor(index / 12), month: (index % 12) + 1 };
}

/**
 * First and last month containing an available date — the calendar never
 * navigates outside this range, so it can't wander into months where nothing
 * is selectable.
 * @param {string[]} dates - ascending ISO dates
 * @returns {{first: YearMonth, last: YearMonth}|null}
 */
export function monthRange(dates) {
  if (!dates.length) return null;
  return { first: monthOf(dates[0]), last: monthOf(dates[dates.length - 1]) };
}

/** @param {YearMonth} ym @param {{first: YearMonth, last: YearMonth}|null} range */
export function clampMonth(ym, range) {
  if (!range) return ym;
  if (compareMonths(ym, range.first) < 0) return range.first;
  if (compareMonths(ym, range.last) > 0) return range.last;
  return ym;
}

export function canShiftMonth(ym, delta, range) {
  if (!range) return false;
  const target = shiftMonth(ym, delta);
  return compareMonths(target, range.first) >= 0 && compareMonths(target, range.last) <= 0;
}

/** Distinct years that contain at least one available date, ascending. */
export function yearsWithDates(dates) {
  return [...new Set(dates.map((d) => Number(d.slice(0, 4))))].sort((a, b) => a - b);
}

/**
 * Which month the calendar should open on: the selected date's month, else
 * the latest available month.
 * @param {string|null} selectedDate
 * @param {string[]} dates
 * @returns {YearMonth|null}
 */
export function initialMonth(selectedDate, dates) {
  if (selectedDate && isIsoDate(selectedDate)) return monthOf(selectedDate);
  return dates.length ? monthOf(dates[dates.length - 1]) : null;
}

// ---------------------------------------------------------------------------
// Month grid
// ---------------------------------------------------------------------------

/**
 * @typedef {object} DayCell
 * @property {string} iso
 * @property {number} day
 * @property {boolean} available - the backend lists this date
 * @property {boolean} selected
 */

/**
 * Builds a Monday-first month view: an array of weeks, each exactly 7 entries,
 * `null` padding the days that belong to neighbouring months.
 * @param {YearMonth} ym
 * @param {ReadonlySet<string>} availableSet
 * @param {string|null} selectedDate
 * @returns {(DayCell|null)[][]}
 */
export function buildMonthGrid(ym, availableSet, selectedDate) {
  const total = daysInMonth(ym.year, ym.month);
  const lead = weekdayIndex(formatIso(ym.year, ym.month, 1));
  const cells = Array.from({ length: lead }, () => null);
  for (let day = 1; day <= total; day += 1) {
    const iso = formatIso(ym.year, ym.month, day);
    cells.push({ iso, day, available: availableSet.has(iso), selected: iso === selectedDate });
  }
  while (cells.length % 7 !== 0) cells.push(null);
  const weeks = [];
  for (let i = 0; i < cells.length; i += 7) weeks.push(cells.slice(i, i + 7));
  return weeks;
}

/** Number of available dates in a month (for the header hint / empty month copy). */
export function countAvailableInMonth(ym, availableSet) {
  let count = 0;
  const total = daysInMonth(ym.year, ym.month);
  for (let day = 1; day <= total; day += 1) {
    if (availableSet.has(formatIso(ym.year, ym.month, day))) count += 1;
  }
  return count;
}

// ---------------------------------------------------------------------------
// Calendar keyboard navigation
// ---------------------------------------------------------------------------

/**
 * Where keyboard focus moves from `focusIso` for a key press, or the same date
 * when the move would leave the navigable range (first day of the first month
 * to last day of the last month). Focus may land on an UNAVAILABLE day — it is
 * announced as disabled and cannot be selected — because skipping over gaps
 * would make arrow-key movement unpredictable.
 *
 * @param {string} focusIso
 * @param {string} key - KeyboardEvent.key
 * @param {{first: YearMonth, last: YearMonth}|null} range
 * @param {boolean} [shiftKey] - with PageUp/PageDown, moves by a year
 * @returns {string}
 */
export function moveCalendarFocus(focusIso, key, range, shiftKey = false) {
  let next;
  switch (key) {
    case "ArrowLeft":
      next = addDays(focusIso, -1);
      break;
    case "ArrowRight":
      next = addDays(focusIso, 1);
      break;
    case "ArrowUp":
      next = addDays(focusIso, -7);
      break;
    case "ArrowDown":
      next = addDays(focusIso, 7);
      break;
    case "Home":
      next = addDays(focusIso, -weekdayIndex(focusIso));
      break;
    case "End":
      next = addDays(focusIso, 6 - weekdayIndex(focusIso));
      break;
    case "PageUp":
      next = addMonths(focusIso, shiftKey ? -12 : -1);
      break;
    case "PageDown":
      next = addMonths(focusIso, shiftKey ? 12 : 1);
      break;
    default:
      return focusIso;
  }
  if (!range) return focusIso;
  const lower = formatIso(range.first.year, range.first.month, 1);
  const upper = formatIso(range.last.year, range.last.month, daysInMonth(range.last.year, range.last.month));
  return next < lower || next > upper ? focusIso : next;
}

/** The day that should receive focus when the calendar opens. */
export function initialFocusDate(selectedDate, month, availableSet) {
  if (selectedDate && monthOfEquals(selectedDate, month)) return selectedDate;
  for (let day = 1; day <= daysInMonth(month.year, month.month); day += 1) {
    const iso = formatIso(month.year, month.month, day);
    if (availableSet.has(iso)) return iso;
  }
  return formatIso(month.year, month.month, 1);
}

function monthOfEquals(iso, ym) {
  const m = monthOf(iso);
  return m.year === ym.year && m.month === ym.month;
}

// ---------------------------------------------------------------------------
// Depth slider — positions are INDICES into the available depth list
// ---------------------------------------------------------------------------
//
// NEER's depth levels (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500,
// 700, 1000 m) are far from evenly spaced. A slider over the raw metre value
// would put 700 → 1000 m across 30% of the track and 0 → 30 m across 3%, and
// could land between levels. The slider therefore moves over 0…n-1 and each
// position maps to exactly one real level.

/** @param {number[]} depths @param {number|null} depth @returns {number} -1 when absent */
export function depthToIndex(depths, depth) {
  return depth === null || depth === undefined ? -1 : depths.indexOf(depth);
}

/** @param {number[]} depths @param {number} index — clamped and rounded @returns {number|null} */
export function indexToDepth(depths, index) {
  if (!depths.length || !Number.isFinite(index)) return null;
  return depths[Math.min(depths.length - 1, Math.max(0, Math.round(index)))];
}

/**
 * The depth an arrow/Home/End key selects from the chip group (radio-group
 * keyboard model: arrows move AND select, wrapping is off so the ends are
 * clear). Returns the current depth for any other key.
 * @param {number[]} depths
 * @param {number|null} current
 * @param {string} key
 * @returns {number|null}
 */
export function navigateDepth(depths, current, key) {
  if (!depths.length) return null;
  const i = depths.indexOf(current);
  switch (key) {
    case "ArrowRight":
    case "ArrowDown":
      return depths[Math.min(depths.length - 1, i === -1 ? 0 : i + 1)];
    case "ArrowLeft":
    case "ArrowUp":
      return depths[Math.max(0, i === -1 ? 0 : i - 1)];
    case "Home":
      return depths[0];
    case "End":
      return depths[depths.length - 1];
    default:
      return current;
  }
}

/** "0 m", "50 m", "1,000 m" — scientific unit always shown. */
export function formatDepthMetres(depth) {
  if (typeof depth !== "number" || !Number.isFinite(depth)) return "--";
  return `${depth.toLocaleString("en-US")} m`;
}

/** Spoken form for assistive tech: "Surface, 0 metres" / "50 metres". */
export function describeDepth(depth) {
  if (typeof depth !== "number" || !Number.isFinite(depth)) return "no depth";
  const unit = depth === 1 ? "metre" : "metres";
  return depth === 0 ? `Surface, 0 ${unit}` : `${depth} ${unit}`;
}