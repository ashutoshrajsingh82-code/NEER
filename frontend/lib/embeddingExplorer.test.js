import { describe, expect, it } from "vitest";
import {
  colorDomain,
  continuousColor,
  getEmbeddingColor,
  monthOfDate,
  normalizeEmbeddingDataset,
  normalizeEmbeddingResponse,
} from "./embeddingExplorer.js";

const vector = (dim = 256) => Array.from({ length: dim }, (_, index) => index / dim);
const response = (date, embedding = vector(), extra = {}) => ({ date, embedding, dim: embedding.length, data_mode: "REAL", ...extra });

describe("embedding response normalization", () => {
  it("accepts the backend date-keyed 256D vector without fabricating metadata", () => {
    const result = normalizeEmbeddingResponse(response("2024-01-31"), "2024-01-31");
    expect(result.ok).toBe(true);
    expect(result.value.dimension).toBe(256);
    expect(result.value.embedding).toHaveLength(256);
    expect(result.value.latitude).toBeNull();
    expect(result.value.longitude).toBeNull();
    expect(result.value.anomaly).toBeNull();
    expect(result.value.region).toBeNull();
  });

  it("rejects invalid shape, wrong date, and non-finite vector entries", () => {
    expect(normalizeEmbeddingResponse({ ...response("2024-01-01"), dim: 255 }, "2024-01-01").ok).toBe(false);
    expect(normalizeEmbeddingResponse(response("2024-01-02"), "2024-01-01").ok).toBe(false);
    const nonFinite = vector(); nonFinite[12] = Infinity;
    expect(normalizeEmbeddingResponse(response("2024-01-01", nonFinite), "2024-01-01").ok).toBe(false);
    expect(normalizeEmbeddingResponse({ date: "2024-01-01", embedding: null, dim: 256 }, "2024-01-01").ok).toBe(false);
  });

  it("reports a real non-256 backend dimension rather than padding it", () => {
    const result = normalizeEmbeddingDataset([response("2024-01-01", vector(128))], ["2024-01-01"]);
    expect(result.dimension).toBe(128);
    expect(result.expectedDimensionMismatch).toBe(true);
    expect(result.records[0].embedding).toHaveLength(128);
  });
});

describe("embedding metadata coloring", () => {
  it("derives month from date-only text without timezone conversion", () => {
    const record = { date: "2024-01-31" };
    expect(monthOfDate(record.date)).toBe(1);
    expect(getEmbeddingColor(record, "month")).toBe(1);
  });

  it("leaves absent latitude, anomaly, and region unavailable", () => {
    const record = { date: "2024-05-01", latitude: null, anomaly: null, region: null };
    expect(getEmbeddingColor(record, "latitude")).toBeNull();
    expect(getEmbeddingColor(record, "anomaly")).toBeNull();
    expect(getEmbeddingColor(record, "region")).toBeNull();
    expect(colorDomain([record], "anomaly")).toBeNull();
  });

  it("centers anomaly visualization domain symmetrically on zero", () => {
    const domain = colorDomain([{ anomaly: -2 }, { anomaly: 5 }], "anomaly");
    expect(domain).toEqual([-5, 5]);
    expect(continuousColor(0, domain, "anomaly")).not.toBe(continuousColor(-5, domain, "anomaly"));
  });
});
