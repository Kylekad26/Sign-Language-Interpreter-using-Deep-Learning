import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Header from "./Header";
import Footer from "./Footer";
import SignLanguageMain from "./SignLanguageMain";
import About from "./About";
import Credits from "./Credits";

export default function App() {
  return (
    <Router>
      <div className="flex flex-col min-h-screen bg-gradient-to-tr from-indigo-100 to-blue-50">
        <Header />
        <main className="flex-grow px-4 md:px-8 py-8">
          <Routes>
            <Route path="/" element={<SignLanguageMain />} />
            <Route path="/about" element={<About />} />
            <Route path="/credits" element={<Credits />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}
