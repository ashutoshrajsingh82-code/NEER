export default function Home() {
  return (
    <main
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        gap: "0.75rem",
        padding: "2rem",
      }}
    >
      <h1 style={{ fontSize: "2.5rem", margin: 0 }}>NEER</h1>
      <p style={{ fontSize: "1.1rem", maxWidth: 480, margin: 0, color: "#444" }}>
        Neural Embedding based Estimation and Reconstruction
      </p>
      <div style={{ marginTop: "1.5rem", fontSize: "0.95rem", color: "#666" }}>
        <p style={{ margin: 0 }}>SIH26066</p>
        <p style={{ margin: 0 }}>MoES / INCOIS</p>
      </div>
    </main>
  );
}
