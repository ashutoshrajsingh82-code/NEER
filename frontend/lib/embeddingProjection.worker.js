const postProgress = (stage, progress) => self.postMessage({ type: "progress", stage, progress });

function centeredMatrix(matrix) {
  const n = matrix.length;
  const d = matrix[0].length;
  const means = new Float64Array(d);
  for (const row of matrix) for (let j = 0; j < d; j += 1) means[j] += row[j] / n;
  let totalVariance = 0;
  const centered = matrix.map((row) => row.map((value, j) => {
    const result = value - means[j];
    totalVariance += result * result;
    return result;
  }));
  return { centered, totalVariance };
}

function eigenvector(gram, previous) {
  const n = gram.length;
  let vector = Float64Array.from({ length: n }, (_, i) => Math.sin((i + 1) * 1.61803398875) + 0.3);
  const normalize = () => {
    if (previous) {
      let dot = 0;
      for (let i = 0; i < n; i += 1) dot += vector[i] * previous[i];
      for (let i = 0; i < n; i += 1) vector[i] -= dot * previous[i];
    }
    let norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    if (!Number.isFinite(norm) || norm < 1e-14) return false;
    for (let i = 0; i < n; i += 1) vector[i] /= norm;
    return true;
  };
  normalize();
  for (let iteration = 0; iteration < 180; iteration += 1) {
    const next = new Float64Array(n);
    for (let i = 0; i < n; i += 1) {
      let sum = 0;
      for (let j = 0; j < n; j += 1) sum += gram[i][j] * vector[j];
      next[i] = sum;
    }
    vector = next;
    if (!normalize()) break;
  }
  let eigenvalue = 0;
  for (let i = 0; i < n; i += 1) {
    let projected = 0;
    for (let j = 0; j < n; j += 1) projected += gram[i][j] * vector[j];
    eigenvalue += vector[i] * projected;
  }
  return { vector, eigenvalue: Math.max(0, eigenvalue) };
}

function pca(matrix) {
  postProgress("Centering embeddings…", 0.08);
  const { centered, totalVariance } = centeredMatrix(matrix);
  const n = centered.length;
  const dimension = centered[0].length;
  let first;
  let second;
  let coordinates;
  if (n <= dimension) {
    // The sample Gram matrix is smaller for short time series and has the
    // same non-zero eigenvalues as the feature covariance matrix.
    const gram = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i += 1) {
      for (let j = i; j < n; j += 1) {
        let sum = 0;
        for (let k = 0; k < dimension; k += 1) sum += centered[i][k] * centered[j][k];
        gram[i][j] = sum;
        gram[j][i] = sum;
      }
    }
    postProgress("Computing principal components…", 0.4);
    first = eigenvector(gram, null);
    second = eigenvector(gram, first.vector);
    coordinates = centered.map((_, i) => [first.vector[i] * Math.sqrt(first.eigenvalue), second.vector[i] * Math.sqrt(second.eigenvalue)]);
  } else {
    // Bound memory for longer histories: covariance is at most 256 × 256,
    // regardless of the number of backend observations.
    const covariance = Array.from({ length: dimension }, () => new Float64Array(dimension));
    for (let a = 0; a < dimension; a += 1) for (let b = a; b < dimension; b += 1) {
      let sum = 0;
      for (let row = 0; row < n; row += 1) sum += centered[row][a] * centered[row][b];
      covariance[a][b] = sum;
      covariance[b][a] = sum;
    }
    postProgress("Computing principal components…", 0.55);
    first = eigenvector(covariance, null);
    second = eigenvector(covariance, first.vector);
    coordinates = centered.map((row) => {
      let x = 0; let y = 0;
      for (let j = 0; j < dimension; j += 1) {
        x += row[j] * first.vector[j];
        y += row[j] * second.vector[j];
      }
      return [x, y];
    });
  }
  return {
    coordinates,
    explainedVariance: totalVariance > 0 ? [first.eigenvalue / totalVariance, second.eigenvalue / totalVariance] : [0, 0],
    axes: ["PC1", "PC2"],
  };
}

function seededRandom(seed) {
  let state = (Number(seed) >>> 0) || 1;
  return () => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}

function pairwiseSquaredDistances(matrix) {
  const n = matrix.length;
  const distances = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      let distance = 0;
      for (let k = 0; k < matrix[i].length; k += 1) {
        const delta = matrix[i][k] - matrix[j][k];
        distance += delta * delta;
      }
      distances[i][j] = distance;
      distances[j][i] = distance;
    }
  }
  return distances;
}

function conditionalProbabilities(distances, perplexity) {
  const n = distances.length;
  const probabilities = Array.from({ length: n }, () => new Float64Array(n));
  const target = Math.log(perplexity);
  for (let i = 0; i < n; i += 1) {
    let beta = 1;
    let lower = -Infinity;
    let upper = Infinity;
    for (let iteration = 0; iteration < 60; iteration += 1) {
      let sum = 0;
      let weighted = 0;
      const row = new Float64Array(n);
      for (let j = 0; j < n; j += 1) {
        if (i === j) continue;
        const value = Math.exp(-distances[i][j] * beta);
        row[j] = value;
        sum += value;
        weighted += distances[i][j] * value;
      }
      if (sum < 1e-300) break;
      const entropy = Math.log(sum) + beta * weighted / sum;
      const delta = entropy - target;
      if (Math.abs(delta) < 1e-6) {
        for (let j = 0; j < n; j += 1) probabilities[i][j] = row[j] / sum;
        break;
      }
      if (delta > 0) {
        lower = beta;
        beta = Number.isFinite(upper) ? (beta + upper) / 2 : beta * 2;
      } else {
        upper = beta;
        beta = Number.isFinite(lower) ? (beta + lower) / 2 : beta / 2;
      }
      if (iteration === 59) for (let j = 0; j < n; j += 1) probabilities[i][j] = row[j] / sum;
    }
  }
  return probabilities;
}

async function tsne(matrix, { perplexity, iterations, seed }) {
  const n = matrix.length;
  postProgress("Calculating pairwise embedding distances…", 0.03);
  const distances = pairwiseSquaredDistances(matrix);
  const conditional = conditionalProbabilities(distances, perplexity);
  const p = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => i === j ? 0 : Math.max(1e-12, (conditional[i][j] + conditional[j][i]) / (2 * n))));
  const random = seededRandom(seed);
  const coordinates = Array.from({ length: n }, () => [(random() - 0.5) * 1e-4, (random() - 0.5) * 1e-4]);
  const velocity = Array.from({ length: n }, () => [0, 0]);
  const gains = Array.from({ length: n }, () => [1, 1]);
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const qNumerator = Array.from({ length: n }, () => new Float64Array(n));
    let qSum = 0;
    for (let i = 0; i < n; i += 1) for (let j = i + 1; j < n; j += 1) {
      const dx = coordinates[i][0] - coordinates[j][0];
      const dy = coordinates[i][1] - coordinates[j][1];
      const q = 1 / (1 + dx * dx + dy * dy);
      qNumerator[i][j] = q; qNumerator[j][i] = q; qSum += 2 * q;
    }
    const gradient = Array.from({ length: n }, () => [0, 0]);
    const exaggeration = iteration < 100 ? 4 : 1;
    for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) {
      if (i === j) continue;
      const q = qNumerator[i][j] / qSum;
      const multiplier = 4 * (exaggeration * p[i][j] - q) * qNumerator[i][j];
      gradient[i][0] += multiplier * (coordinates[i][0] - coordinates[j][0]);
      gradient[i][1] += multiplier * (coordinates[i][1] - coordinates[j][1]);
    }
    const momentum = iteration < 250 ? 0.5 : 0.8;
    for (let i = 0; i < n; i += 1) for (let axis = 0; axis < 2; axis += 1) {
      gains[i][axis] = Math.sign(gradient[i][axis]) !== Math.sign(velocity[i][axis]) ? gains[i][axis] + 0.2 : Math.max(0.01, gains[i][axis] * 0.8);
      velocity[i][axis] = momentum * velocity[i][axis] - 200 * gains[i][axis] * gradient[i][axis];
      coordinates[i][axis] += velocity[i][axis];
    }
    if (iteration % 25 === 0) {
      postProgress("Computing t-SNE…", (iteration + 1) / iterations);
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  return { coordinates, explainedVariance: null, axes: ["t-SNE 1", "t-SNE 2"] };
}

self.onmessage = async (event) => {
  const { id, method, matrix, parameters } = event.data || {};
  try {
    if (!Array.isArray(matrix) || matrix.length < 2 || !matrix.every((row) => Array.isArray(row) && row.length > 0 && row.every(Number.isFinite))) {
      throw new Error("At least two complete finite embedding vectors are required.");
    }
    const result = method === "pca" ? pca(matrix) : await tsne(matrix, parameters || {});
    self.postMessage({ type: "result", id, result });
  } catch (error) {
    self.postMessage({ type: "error", id, message: error instanceof Error ? error.message : String(error) });
  }
};
