
import { Link, useNavigate } from "react-router-dom";
import "../App.css";

function Dashboard() {
  const navigate = useNavigate();

  const cardStyle = {
    background: "#ffffff",
    borderRadius: "16px",
    padding: "24px",
    boxShadow: "0 6px 20px rgba(0,0,0,0.06)",
  };

  const buttonStyle = {
    display: "inline-block",
    marginTop: "16px",
    padding: "12px 20px",
    background: "#2563eb",
    color: "white",
    borderRadius: "8px",
    textDecoration: "none",
    fontWeight: "600",
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/login");
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f1f5f9",
        padding: "40px",
        fontFamily: "Arial, sans-serif",
      }}
    >
      {/* Header */}
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "16px",
          marginBottom: "40px",
        }}
      >
        <div>
          <h1 style={{ color: "#0f172a", margin: 0 }}>
            QuantumHealth
          </h1>
          <p style={{ color: "#64748b" }}>
            Hybrid Quantum-Classical Health Assessment
          </p>
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "12px",
            flexWrap: "wrap",
          }}
        >
          <Link
            to="/history"
            style={{
              ...buttonStyle,
              marginTop: 0,
            }}
          >
            Prediction History
          </Link>

          <Link
            to="/performance"
            style={{
              ...buttonStyle,
              marginTop: 0,
              background: "#0f766e",
            }}
          >
            Model Performance
          </Link>

          <button
            onClick={handleLogout}
            style={{
              ...buttonStyle,
              marginTop: 0,
              border: "none",
              cursor: "pointer",
            }}
          >
            Logout
          </button>
        </div>
      </header>

      {/* Welcome */}
      <section style={{ marginBottom: "32px" }}>
        <h2 style={{ color: "#0f172a" }}>
          Welcome to your Dashboard
        </h2>

        <p style={{ color: "#64748b" }}>
          Select a health assessment to estimate your disease risk.
        </p>
      </section>

      {/* Disease Cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
          gap: "24px",
        }}
      >
        {/* Diabetes */}
        <div style={cardStyle}>
          <h2 style={{ color: "#1e293b" }}>
            Diabetes
          </h2>

          <p style={{ color: "#64748b" }}>
            Estimate diabetes risk using health measurements such as glucose,
            BMI, age and blood pressure.
          </p>

          <Link to="/diabetes" style={buttonStyle}>
            Start Assessment
          </Link>
        </div>

        {/* Heart Disease */}
        <div style={cardStyle}>
          <h2 style={{ color: "#1e293b" }}>
            Heart Disease
          </h2>

          <p style={{ color: "#64748b" }}>
            Assess heart disease risk using relevant cardiovascular health
            information.
          </p>

          <Link to="/heart" style={buttonStyle}>
            Start Assessment
          </Link>
        </div>

        {/* Pneumonia */}
        <div style={cardStyle}>
          <h2 style={{ color: "#1e293b" }}>
            Pneumonia
          </h2>

          <p style={{ color: "#64748b" }}>
            Upload a chest X-ray image to estimate pneumonia risk.
          </p>

          <Link to="/pneumonia" style={buttonStyle}>
            Start Assessment
          </Link>
        </div>
      </div>

      {/* Information */}
      <div
        style={{
          marginTop: "40px",
          padding: "20px",
          background: "#e0f2fe",
          borderRadius: "12px",
          color: "#075985",
        }}
      >
        <strong>Note:</strong> This platform is a student research prototype
        for educational purposes. It does not provide a medical diagnosis.
      </div>
    </div>
  );
}

export default Dashboard;
