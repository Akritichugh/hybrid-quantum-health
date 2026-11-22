
import { Link, useNavigate } from "react-router-dom";
import "../App.css";
import { useState } from "react";
import api from "../api/axios";

function Login() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");

  const handleLogin = async (e) => {
    e.preventDefault();

    try {
      const response = await api.post("/api/auth/login", {
        email: email,
        password: password,
      });

      // Save the access token for authenticated API requests
      localStorage.setItem(
        "token",
        response.data.access_token
      );

      // Save user information
      localStorage.setItem(
        "user",
        JSON.stringify(response.data.user)
      );

      setMessage("Login successful!");

      navigate("/dashboard");
    } catch (error) {
      console.error("Login error:", error);

      setMessage(
        error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message ||
        "Login failed"
      );
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <h1>Welcome Back</h1>

        <p>Login to your QuantumHealth account</p>

        <form onSubmit={handleLogin}>
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
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <button type="submit" className="primary-btn">
            Login
          </button>

          {message && <p>{message}</p>}
        </form>

        <p className="auth-footer">
          Don't have an account?{" "}
          <Link to="/register">Register</Link>
        </p>
      </div>
    </div>
  );
}

export default Login;
