import React, { createContext, useState, useEffect } from 'react';

// Create the context here instead of importing it
export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check for saved user data in localStorage on initial load
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      setCurrentUser(JSON.parse(savedUser));
    }
    setLoading(false);
  }, []);

  // Mock signup function
  const signup = async (name, email, password) => {
    try {
      // Here you would typically call your backend API
      console.log("Signing up user:", name, email);
      
      // Mock successful signup
      const userData = { name, email };
      localStorage.setItem('user', JSON.stringify(userData));
      setCurrentUser(userData);
      return true;
    } catch (error) {
      console.error("Signup error:", error);
      return false;
    }
  };

  // Mock login function
  const login = async (email, password) => {
    try {
      // Mock successful login
      const userData = { email };
      localStorage.setItem('user', JSON.stringify(userData));
      setCurrentUser(userData);
      return true;
    } catch (error) {
      console.error("Login error:", error);
      return false;
    }
  };

  // Mock logout function
  const logout = () => {
    localStorage.removeItem('user');
    setCurrentUser(null);
    return true;
  };

  const value = {
    currentUser,
    loading,
    signup,
    login,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
};