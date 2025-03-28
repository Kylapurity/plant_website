import React, { useState, useContext, useRef, useEffect } from "react";
import { AuthContext } from "../AuthContext";
import { getRecommendations } from "./recommendations";

const Dashboard = () => {
  const { token, logout } = useContext(AuthContext);
  const [activeTab, setActiveTab] = useState("predict");
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [retrainingHistory, setRetrainingHistory] = useState([]);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState(null);
  const [error, setError] = useState(null);
  const [trainingFiles, setTrainingFiles] = useState([]);
  const [isRetraining, setIsRetraining] = useState(false);
  const fileInputRef = useRef(null);
  const trainingFileInputRef = useRef(null);

  const API_URL = "http://127.0.0.1:8000";

  useEffect(() => {
    if (token) {
      fetchPredictionHistory();
      fetchRetrainingHistory();
    } else {
      setError("You must be logged in to access this page.");
    }
  }, [token]);

  const fetchPredictionHistory = async () => {
    try {
      const response = await fetch(`${API_URL}/prediction_history`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!response.ok) throw new Error("Failed to fetch prediction history");
      const data = await response.json();
      setPredictionHistory(data);
    } catch (err) {
      setError(err.message);
      console.error("Error fetching prediction history:", err);
    }
  };

  const fetchRetrainingHistory = async () => {
    try {
      const response = await fetch(`${API_URL}/retraining_history`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!response.ok) throw new Error("Failed to fetch retraining history");
      const data = await response.json();
      setRetrainingHistory(data);
    } catch (err) {
      setError(err.message);
      console.error("Error fetching retraining history:", err);
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnalysisResult(null);
      setError(null);
    }
  };

  const handleAnalyzeClick = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedImage);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Prediction failed");
      }

      const data = await response.json();
      const rawDiseaseName = data.prediction;
      const recommendations = getRecommendations(rawDiseaseName); // Fetch recommendations

      setAnalysisResult({
        disease: rawDiseaseName,
        rawDiseaseName: rawDiseaseName,
        confidence: Math.round(data.confidence * 100),
        timestamp: data.timestamp,
        recommendations: recommendations, // Add recommendations to result
      });

      await fetchPredictionHistory();
    } catch (err) {
      setError(err.message || "Failed to analyze the image. Please try again.");
      console.error("Error during prediction:", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleRetrainSubmit = async (e) => {
    e.preventDefault();
    if (!trainingFiles.length) {
      setError("Please select files for retraining");
      return;
    }

    setIsRetraining(true);
    setError(null);

    try {
      const formData = new FormData();
      trainingFiles.forEach((file) => formData.append("files", file));

      const response = await fetch(`${API_URL}/retrain`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Retraining failed");
      }

      const data = await response.json();
      alert("Retraining completed successfully!");
      setTrainingFiles([]);
      trainingFileInputRef.current.value = null;

      await fetchRetrainingHistory();
    } catch (err) {
      setError(err.message || "Retraining failed. Please try again.");
      console.error("Error during retraining:", err);
    } finally {
      setIsRetraining(false);
    }
  };

  const handleTrainingFileChange = (e) => {
    setTrainingFiles([...e.target.files]);
  };

  const viewHistoryDetails = (item) => setSelectedHistoryItem(item);
  const closeHistoryDetails = () => setSelectedHistoryItem(null);
  const handleNewScan = () => {
    setSelectedImage(null);
    setPreviewUrl("");
    setAnalysisResult(null);
    setError(null);
    setActiveTab("predict");
  };

  const navTabs = [
    { id: "predict", label: "Predict Disease" },
    { id: "prediction_history", label: "Prediction History" },
    { id: "retrain", label: "Retrain Model" },
    { id: "retraining_history", label: "Retraining History" },
  ];

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <h1 className="text-xl font-bold text-green-600">
                  Plant Disease Detector
                </h1>
              </div>
            </div>
            <div className="flex items-center">
              <button
                onClick={logout}
                className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="mb-4 border-b border-gray-200">
            <ul className="flex flex-wrap -mb-px">
              {navTabs.map((tab) => (
                <li key={tab.id} className="mr-2">
                  <button
                    className={`inline-block py-4 px-4 text-sm font-medium ${
                      activeTab === tab.id
                        ? "text-green-600 border-b-2 border-green-600"
                        : "text-gray-500 hover:text-gray-700"
                    }`}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    {tab.label}
                  </button>
                </li>
              ))}
            </ul>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg">
              {error}
              <button
                className="ml-4 bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-4 rounded"
                onClick={() => setError(null)}
              >
                Dismiss
              </button>
            </div>
          )}

          {activeTab === "predict" && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Predict Plant Disease</h2>
              <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
                {!previewUrl ? (
                  <div className="flex flex-col items-center">
                    <div className="flex items-center justify-center w-full mb-4">
                      <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                          <svg
                            className="w-10 h-10 mb-3 text-gray-400"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth="2"
                              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                            ></path>
                          </svg>
                          <p className="mb-2 text-sm text-gray-500">
                            <span className="font-semibold">
                              Click to upload
                            </span>{" "}
                            or drag and drop
                          </p>
                          <p className="text-xs text-gray-500">
                            PNG, JPG, or JPEG (MAX. 10MB)
                          </p>
                        </div>
                        <input
                          id="dropzone-file"
                          type="file"
                          className="hidden"
                          accept="image/*"
                          ref={fileInputRef}
                          onChange={handleImageChange}
                        />
                      </label>
                    </div>
                    <button
                      type="button"
                      className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded"
                      onClick={() => {
                        if (!selectedImage) {
                          alert("Please upload an image first");
                          fileInputRef.current.click();
                        } else {
                          handleAnalyzeClick();
                        }
                      }}
                    >
                      Predict Plant Disease
                    </button>
                  </div>
                ) : (
                  <div className="flex flex-col items-center">
                    <div className="relative mb-4 w-full max-w-lg">
                      <img
                        src={previewUrl}
                        alt="Plant preview"
                        className="rounded-lg shadow-md object-contain max-h-64 mx-auto"
                      />
                      <button
                        onClick={() => {
                          setSelectedImage(null);
                          setPreviewUrl("");
                          setAnalysisResult(null);
                          setError(null);
                        }}
                        className="absolute top-2 right-2 bg-red-500 hover:bg-red-700 text-white rounded-full p-1"
                      >
                        <svg
                          className="w-5 h-5"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M6 18L18 6M6 6l12 12"
                          ></path>
                        </svg>
                      </button>
                    </div>

                    {!analysisResult && (
                      <button
                        type="button"
                        className={`bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded ${
                          isAnalyzing ? "opacity-75 cursor-not-allowed" : ""
                        }`}
                        onClick={handleAnalyzeClick}
                        disabled={isAnalyzing}
                      >
                        {isAnalyzing ? "Analyzing..." : "Predict Disease"}
                      </button>
                    )}
                  </div>
                )}

                {analysisResult && (
                  <div className="mt-8 border-t pt-6">
                    <h3 className="text-xl font-bold mb-4">Analysis Results</h3>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <div className="mb-4">
                        <span className="font-semibold">Predicted Disease:</span>{" "}
                        <span
                          className={
                            analysisResult.disease.toLowerCase().includes("healthy")
                              ? "text-green-600"
                              : "text-red-600"
                          }
                        >
                          {analysisResult.disease}
                        </span>
                      </div>
                      <div className="mb-4">
                        <span className="font-semibold">Confidence:</span>{" "}
                        <span>{analysisResult.confidence}%</span>
                      </div>
                      <div className="mb-4">
                        <span className="font-semibold">Timestamp:</span>{" "}
                        <span>
                          {new Date(analysisResult.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <div className="mb-6">
                        <span className="font-semibold">Recommendations:</span>
                        <ul className="list-disc pl-5 mt-2">
                          {analysisResult.recommendations.map((rec, index) => (
                            <li key={index} className="text-gray-700">
                              {rec}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div className="flex justify-center">
                        <button
                          type="button"
                          className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded"
                          onClick={handleNewScan}
                        >
                          Scan Another Plant
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "prediction_history" && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Prediction History</h2>
              <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
                {predictionHistory.length === 0 ? (
                  <div>
                    <p className="text-gray-600 mb-4 text-center">
                      You haven't made any predictions yet.
                    </p>
                    <div className="flex justify-center">
                      <button
                        type="button"
                        className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded"
                        onClick={handleNewScan}
                      >
                        Make a Prediction
                      </button>
                    </div>
                  </div>
                ) : (
                  <div>
                    {selectedHistoryItem ? (
                      <div className="bg-white p-4 rounded-lg">
                        <div className="flex justify-between items-start mb-4">
                          <h3 className="text-xl font-bold">
                            {selectedHistoryItem.text.split(": ")[1]}
                          </h3>
                          <button
                            onClick={closeHistoryDetails}
                            className="text-gray-500 hover:text-gray-700"
                          >
                            <svg
                              className="w-6 h-6"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth="2"
                                d="M6 18L18 6M6 6l12 12"
                              ></path>
                            </svg>
                          </button>
                        </div>
                        <div className="mb-4">
                          <span className="font-semibold">Confidence:</span>{" "}
                          <span>
                            {Math.round(selectedHistoryItem.confidence * 100)}%
                          </span>
                        </div>
                        <div className="mb-4">
                          <span className="font-semibold">Date:</span>{" "}
                          <span>
                            {new Date(
                              selectedHistoryItem.date
                            ).toLocaleString()}
                          </span>
                        </div>
                        <div className="mt-6 flex justify-center">
                          <button
                            type="button"
                            className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded"
                            onClick={handleNewScan}
                          >
                            New Prediction
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                          {predictionHistory.map((item) => (
                            <div
                              key={item.id}
                              className="bg-white overflow-hidden shadow-md rounded-lg hover:shadow-lg transition-shadow cursor-pointer"
                              onClick={() => viewHistoryDetails(item)}
                            >
                              <div className="p-4">
                                <p className="text-gray-500 text-sm">
                                  {new Date(item.date).toLocaleDateString()}
                                </p>
                                <h3
                                  className={`font-bold mt-1 ${
                                    item.text.toLowerCase().includes("healthy")
                                      ? "text-green-600"
                                      : "text-red-600"
                                  }`}
                                >
                                  {item.text.split(": ")[1]}
                                </h3>
                                <p className="text-gray-700 mt-1">
                                  Confidence:{" "}
                                  {Math.round(item.confidence * 100)}%
                                </p>
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="mt-6 flex justify-center">
                          <button
                            type="button"
                            className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded"
                            onClick={handleNewScan}
                          >
                            New Prediction
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "retrain" && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Retrain Model</h2>
              <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
                <form onSubmit={handleRetrainSubmit}>
                  <div className="mb-4">
                    <label className="block text-gray-700 mb-2">
                      Upload Training Images
                    </label>
                    <input
                      type="file"
                      ref={trainingFileInputRef}
                      onChange={handleTrainingFileChange}
                      multiple
                      className="w-full p-2 border rounded"
                      accept="image/*,.zip"
                      required
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Upload images or a ZIP file containing images.
                    </p>
                  </div>
                  <button
                    type="submit"
                    disabled={isRetraining}
                    className={`bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded ${
                      isRetraining ? "opacity-75 cursor-not-allowed" : ""
                    }`}
                  >
                    {isRetraining ? "Retraining..." : "Start Retraining"}
                  </button>
                  {trainingFiles.length > 0 && (
                    <p className="mt-2 text-sm text-gray-600">
                      {trainingFiles.length} files selected
                    </p>
                  )}
                </form>
              </div>
            </div>
          )}

          {activeTab === "retraining_history" && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Retraining History</h2>
              <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
                {retrainingHistory.length === 0 ? (
                  <div>
                    <p className="text-gray-600 mb-4 text-center">
                      You haven't retrained the model yet.
                    </p>
                    <div className="flex justify-center">
                      <button
                        type="button"
                        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded"
                        onClick={() => setActiveTab("retrain")}
                      >
                        Retrain Model
                      </button>
                    </div>
                  </div>
                ) : (
                  <div>
                    {selectedHistoryItem ? (
                      <div className="bg-white p-4 rounded-lg">
                        <div className="flex justify-between items-start mb-4">
                          <h3 className="text-xl font-bold">
                            {selectedHistoryItem.text}
                          </h3>
                          <button
                            onClick={closeHistoryDetails}
                            className="text-gray-500 hover:text-gray-700"
                          >
                            <svg
                              className="w-6 h-6"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth="2"
                                d="M6 18L18 6M6 6l12 12"
                              ></path>
                            </svg>
                          </button>
                        </div>
                        <div className="mb-4">
                          <span className="font-semibold">
                            Training Accuracy:
                          </span>{" "}
                          <span>
                            {selectedHistoryItem.training_accuracy
                              ? `${Math.round(
                                  selectedHistoryItem.training_accuracy * 100
                                )}%`
                              : "N/A"}
                          </span>
                        </div>
                        <div className="mb-4">
                          <span className="font-semibold">
                            Validation Accuracy:
                          </span>{" "}
                          <span>
                            {selectedHistoryItem.validation_accuracy
                              ? `${Math.round(
                                  selectedHistoryItem.validation_accuracy * 100
                                )}%`
                              : "N/A"}
                          </span>
                        </div>
                        <div className="mb-4">
                          <span className="font-semibold">Date:</span>{" "}
                          <span>
                            {new Date(
                              selectedHistoryItem.date
                            ).toLocaleString()}
                          </span>
                        </div>
                        {selectedHistoryItem.class_metrics &&
                          Object.keys(selectedHistoryItem.class_metrics)
                            .length > 0 && (
                            <div>
                              <h4 className="font-semibold mb-2">
                                Class Metrics:
                              </h4>
                              <ul className="list-disc pl-5">
                                {Object.entries(
                                  selectedHistoryItem.class_metrics
                                ).map(([className, metrics]) => (
                                  <li
                                    key={className}
                                    className="text-gray-700 mb-1"
                                  >
                                    {className}: Precision:{" "}
                                    {Math.round(metrics.precision * 100)}%,
                                    Recall: {Math.round(metrics.recall * 100)}%,
                                    F1: {Math.round(metrics.f1_score * 100)}%
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        <div className="mt-6 flex justify-center">
                          <button
                            type="button"
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded"
                            onClick={() => setActiveTab("retrain")}
                          >
                            Retrain Again
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                          {retrainingHistory.map((item) => (
                            <div
                              key={item.id}
                              className="bg-white overflow-hidden shadow-md rounded-lg hover:shadow-lg transition-shadow cursor-pointer"
                              onClick={() => viewHistoryDetails(item)}
                            >
                              <div className="p-4">
                                <p className="text-gray-500 text-sm">
                                  {new Date(item.date).toLocaleDateString()}
                                </p>
                                <h3 className="font-bold mt-1 text-gray-900">
                                  {item.text}
                                </h3>
                                <p className="text-gray-700 mt-1">
                                  Accuracy:{" "}
                                  {item.training_accuracy
                                    ? `${Math.round(
                                        item.training_accuracy * 100
                                      )}%`
                                    : "N/A"}
                                </p>
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="mt-6 flex justify-center">
                          <button
                            type="button"
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded"
                            onClick={() => setActiveTab("retrain")}
                          >
                            Retrain Model
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;