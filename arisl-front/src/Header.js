import React from "react";
import { Link, useLocation } from "react-router-dom";

export default function Header() {
  const location = useLocation();
  return (
    <header className="bg-indigo-700 text-white shadow p-4 flex justify-between items-center">
      <div className="text-lg font-extrabold tracking-widest">
        ARISL Sign Language Interpreter
      </div>
      <nav className="flex gap-6 text-lg font-semibold">
        <Link to="/" className={location.pathname === "/" ? "underline decoration-yellow-400" : "hover:underline"}>
          Home
        </Link>
        <Link to="/about" className={location.pathname === "/about" ? "underline decoration-yellow-400" : "hover:underline"}>
          About
        </Link>
        <Link to="/credits" className={location.pathname === "/credits" ? "underline decoration-yellow-400" : "hover:underline"}>
          Credits
        </Link>
      </nav>
    </header>
  );
}
