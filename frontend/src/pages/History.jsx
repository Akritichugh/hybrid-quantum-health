
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import api from "../api/axios";

function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await api.get("/api/predict/history");
        setHistory(response.data);
      } catch (err) {
        console.error("History error:", err);

        if (err.response?.status === 401) {
          setError("Please log in again to view your prediction history.");
        } else {
          setError("Could not load prediction history.");
        }
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  const formatDate = (date) => {
    if (!date) return "Date unavailable";

    return new Date(date).toLocaleString();
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        padding: "30px",
        background: "#f4f7fb",
      }}
    >
      <div
        style={{
          maxWidth: "1000px",
          margin: "0 auto",
        }}
      >
        <Link
          to="/dashboard"
          style={{
            textDecoration: "none",
            color: "#2563eb",
            fontWeight: "600",
          }}
        >
          ← Back to Dashboard
        </Link>

        <h1 style={{ marginTop: "25px" }}>
          Prediction History
        </h1>

        <p style={{ color: "#64748b" }}>
          View your previously saved disease prediction results.
        </p>

        {loading && <p>Loading your history...</p>}

        {error && (
          <div
            style={{
              background: "#fee2e2",
              color: "#991b1b",
              padding: "15px",
              borderRadius: "8px",
              marginTop: "20px",
            }}
          >
            {error}

            {error.includes("log in") && (
              <div style={{ marginTop: "10px" }}>
                <Link to="/login">Go to Login</Link>
              </div>
            )}
          </div>
        )}

        {!loading && !error && history.length === 0 && (
          <div
            style={{
              background: "white",
              padding: "25px",
              borderRadius: "12px",
              marginTop: "25px",
              textAlign: "center",
            }}
          >
            <h3>No predictions yet</h3>
            <p>Your saved predictions will appear here.</p>
            <Link to="/dashboard">Make a Prediction</Link>
          </div>
        )}

        {!loading && !error && history.length > 0 && (
          <div
            style={{
              display: "grid",
              gap: "18px",
              marginTop: "25px",
            }}
          >
            {history.map((item) => (
              <div
                key={item.id}
                style={{
                  background: "white",
                  padding: "22px",
                  borderRadius: "12px",
                  boxShadow: "0 3px 12px rgba(0,0,0,0.06)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    flexWrap: "wrap",
                    gap: "10px",
                  }}
                >
                  <h3 style={{ margin: 0 }}>
                    {item.disease}
                  </h3>

                  <span
                    style={{
                      fontWeight: "600",
                      color: item.prediction === 1 ? "#dc2626" : "#15803d",
                    }}
                  >
                    {item.prediction === 1
                      ? "Positive model result"
                      : "Negative model result"}
                  </span>
                </div>

                <p style={{ marginBottom: "8px" }}>
                  Estimated probability:{" "}
                  <strong>
                    {item.estimated_probability}%
                  </strong>
                </p>

                <p>{item.message}</p>

                <p
                  style={{
                    color: "#64748b",
                    fontSize: "13px",
                    marginBottom: 0,
                  }}
                >
                  {formatDate(item.created_at)}
                </p>
              </div>
            ))}
          </div>
        )}

        <p
          style={{
            color: "#64748b",
            fontSize: "13px",
            marginTop: "25px",
          }}
        >
          Educational research prototype only. These model results
          are not a medical diagnosis.
        </p>
      </div>
    </div>
  );
}

export default History;
