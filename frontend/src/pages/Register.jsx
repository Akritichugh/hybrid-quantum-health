
import { Link, useNavigate } from "react-router-dom";
import "../App.css";
import { useState } from "react";
import api from "../api/axios";

function Register() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  const handleRegister = async (e) => {
    e.preventDefault();
    setMessage("");
    setLoading(true);

    try {
      await api.post("/api/auth/register", {
        full_name: fullName,
        email: email,
        password: password,
      });

      // Log in the newly registered patient
      const loginResponse = await api.post("/api/auth/login", {
        email: email,
        password: password,
      });

      localStorage.setItem(
        "token",
        loginResponse.data.access_token
      );

      localStorage.setItem(
        "user",
        JSON.stringify(loginResponse.data.user)
      );

      navigate("/dashboard");
    } catch (error) {
      console.error("Registration error:", error);

      const detail = error.response?.data?.detail;

if (Array.isArray(detail)) {
  setMessage(detail.map((item) => item.msg).join(", "));
} else if (typeof detail === "object" && detail !== null) {
  setMessage(detail.msg || JSON.stringify(detail));
} else {
  setMessage(
    detail ||
    error.response?.data?.message ||
    error.message ||
    "Registration failed"
  );
}
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <h1>Create Account</h1>
        <p>Join the QuantumHealth platform</p>

        <form onSubmit={handleRegister}>
          <label>Full Name</label>
          <input
            type="text"
            placeholder="Enter your name"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            required
          />

          <label>Email Address</label>
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <label>Password</label>
          <input
            type="password"
            placeholder="Create a password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <button
            type="submit"
            className="primary-btn"
            disabled={loading}
          >
            {loading ? "Creating Account..." : "Create Account"}
          </button>

          {message && <p>{message}</p>}
        </form>

        <p className="auth-footer">
          Already have an account? <Link to="/login">Login</Link>
        </p>
      </div>
    </div>
  );
}

export default Register;
