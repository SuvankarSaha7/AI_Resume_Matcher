import React, { useState } from 'react';
import { Upload, File, X, User, Sparkles, CheckCircle2, FileText, Users } from 'lucide-react';

// Navbar Component
const Navbar = () => {
  return (
    <nav className="bg-gradient-to-r from-slate-900 via-purple-900 to-slate-900 shadow-2xl border-b border-purple-500/20 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-6 py-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl blur-lg opacity-75 animate-pulse"></div>
              <div className="relative w-14 h-14 bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 rounded-2xl flex items-center justify-center shadow-2xl">
                <span className="text-white font-black text-2xl">ITC</span>
              </div>
            </div>
            <div>
              <span className="text-3xl font-black bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Interview Prep
              </span>
              <p className="text-xs text-purple-300 font-medium">AI-Powered Excellence</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full blur opacity-75 group-hover:opacity-100 transition-opacity"></div>
              <div className="relative w-12 h-12 bg-gradient-to-br from-purple-600 to-pink-600 rounded-full flex items-center justify-center cursor-pointer transform group-hover:scale-110 transition-all duration-300 shadow-lg">
                <User className="w-6 h-6 text-white" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

// FileUploadCard Component
const FileUploadCard = ({ title, subtitle, files, onFileUpload, onFileRemove, accept, multiple = false, icon: Icon }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    onFileUpload(droppedFiles);
  };

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    onFileUpload(selectedFiles);
  };

  return (
    <div className="relative group">
      {/* Animated gradient border */}
      <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-3xl blur opacity-30 group-hover:opacity-100 transition duration-500 animate-pulse"></div>
      
      <div className="relative bg-gradient-to-br from-slate-900 to-slate-800 rounded-3xl shadow-2xl p-8 border border-purple-500/20 backdrop-blur-xl">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
            <Icon className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-black text-white">{title}</h2>
            <p className="text-sm text-purple-300">{subtitle}</p>
          </div>
        </div>
        
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`relative border-2 border-dashed rounded-2xl p-10 text-center transition-all duration-500 overflow-hidden ${
            isDragging
              ? 'border-purple-400 bg-purple-900/30 scale-105 shadow-2xl'
              : 'border-slate-600 bg-slate-800/50 hover:bg-slate-700/50 hover:border-purple-500/50'
          }`}
        >
          {/* Animated background gradient */}
          <div className={`absolute inset-0 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 transition-opacity duration-500 ${isDragging ? 'opacity-100' : 'opacity-0'}`}></div>
          
          <input
            type="file"
            id={`file-${title}`}
            className="hidden"
            accept={accept}
            multiple={multiple}
            onChange={handleFileChange}
          />
          <label htmlFor={`file-${title}`} className="cursor-pointer relative z-10">
            <div className="flex flex-col items-center">
              <div className={`relative w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6 shadow-xl transition-all duration-500 ${isDragging ? 'scale-125 rotate-12' : 'group-hover:scale-110'}`}>
                <Upload className="w-10 h-10 text-white" />
                <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-400 rounded-2xl blur-xl opacity-50"></div>
              </div>
              <p className="text-xl font-bold text-white mb-2">
                Drag & drop or click to upload
              </p>
              <p className="text-sm text-purple-300 flex items-center space-x-2">
                <Sparkles className="w-4 h-4" />
                <span>Supports PDF, DOCX, and TXT files</span>
              </p>
            </div>
          </label>
        </div>

        {files.length > 0 && (
          <div className="mt-8 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-purple-300 uppercase tracking-wider flex items-center space-x-2">
                <CheckCircle2 className="w-4 h-4" />
                <span>Uploaded Files ({files.length})</span>
              </h3>
            </div>
            <div className="space-y-3 max-h-64 overflow-y-auto custom-scrollbar">
              {files.map((file, index) => (
                <div
                  key={index}
                  className="relative group/file"
                >
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl blur opacity-0 group-hover/file:opacity-75 transition duration-300"></div>
                  <div className="relative flex items-center justify-between bg-gradient-to-r from-slate-800 to-slate-700 p-4 rounded-xl border border-purple-500/20 hover:border-purple-400/50 transition-all shadow-lg">
                    <div className="flex items-center space-x-4 flex-1 min-w-0">
                      <div className="relative">
                        <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center flex-shrink-0 shadow-lg">
                          <File className="w-6 h-6 text-white" />
                        </div>
                        <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-slate-800 animate-pulse"></div>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-bold text-white truncate">
                          {file.name}
                        </p>
                        <div className="flex items-center space-x-2 mt-1">
                          <span className="text-xs text-purple-300">
                            {(file.size / 1024).toFixed(2)} KB
                          </span>
                          <span className="text-xs text-green-400 flex items-center space-x-1">
                            <CheckCircle2 className="w-3 h-3" />
                            <span>Ready</span>
                          </span>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => onFileRemove(index)}
                      className="ml-4 w-10 h-10 bg-red-500/20 rounded-lg flex items-center justify-center hover:bg-red-500 hover:scale-110 transition-all flex-shrink-0 border border-red-500/30"
                    >
                      <X className="w-5 h-5 text-red-400 hover:text-white transition-colors" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Main Dashboard Component
const Dashboard = () => {
  const [jobDescriptions, setJobDescriptions] = useState([]);
  const [resumes, setResumes] = useState([]);

  const handleJobDescriptionUpload = (newFiles) => {
    if(newFiles.length>0){

      setJobDescriptions([newFiles[0]]);
    }
  };

  const handleResumeUpload = (newFiles) => {
    setResumes([...resumes, ...newFiles]);
  };

  const handleJobDescriptionRemove = (index) => {
    setJobDescriptions(jobDescriptions.filter((_, i) => i !== index));
  };

  const handleResumeRemove = (index) => {
    setResumes(resumes.filter((_, i) => i !== index));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse"></div>
        <div className="absolute top-1/2 right-1/4 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse delay-1000"></div>
        <div className="absolute bottom-1/4 left-1/2 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse delay-2000"></div>
      </div>

      <Navbar />
      
      <div className="relative z-10 max-w-7xl mx-auto px-6 py-16">
        <div className="mb-12 text-center">
          <div className="flex items-center justify-center space-x-3 mb-4">
            <Sparkles className="w-8 h-8 text-purple-400 animate-pulse" />
            <h1 className="text-6xl font-black bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Welcome to Interview Prep
            </h1>
            <Sparkles className="w-8 h-8 text-blue-400 animate-pulse" />
          </div>
          <p className="text-xl text-purple-300 font-medium max-w-3xl mx-auto">
            Upload your job descriptions and resumes to unlock AI-powered interview preparation and personalized insights
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          <FileUploadCard
            title="Job Description"
            subtitle="Upload the job description you're applying for"
            files={jobDescriptions}
            onFileUpload={handleJobDescriptionUpload}
            onFileRemove={handleJobDescriptionRemove}
            accept=".pdf,.docx,.txt,.doc"
            multiple={false}
            icon={FileText}
          />
          
          <FileUploadCard
            title="Resumes"
            subtitle="Upload one or multiple resumes"
            files={resumes}
            onFileUpload={handleResumeUpload}
            onFileRemove={handleResumeRemove}
            accept=".pdf,.docx,.txt,.doc"
            multiple={true}
            icon={Users}
          />
        </div>

        {(jobDescriptions.length > 0 || resumes.length > 0) && (
          <div className="relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-3xl blur opacity-50 group-hover:opacity-100 transition duration-500 animate-pulse"></div>
            <div className="relative bg-gradient-to-br from-slate-900 to-slate-800 rounded-3xl shadow-2xl p-8 border border-purple-500/20 backdrop-blur-xl">
              <div className="flex flex-col lg:flex-row items-center justify-between space-y-4 lg:space-y-0">
                <div className="flex items-center space-x-4">
                  <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center shadow-xl">
                    <CheckCircle2 className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-black text-white flex items-center space-x-2">
                      <span>Ready to proceed?</span>
                      <Sparkles className="w-5 h-5 text-yellow-400 animate-pulse" />
                    </h3>
                    <p className="text-sm text-purple-300 font-medium">
                      {jobDescriptions.length} job description(s) • {resumes.length} resume(s) • AI Analysis Ready
                    </p>
                  </div>
                </div>
                <button className="relative group/btn px-10 py-5 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 text-white font-black text-lg rounded-2xl hover:shadow-2xl transform hover:scale-105 transition-all duration-300 overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 opacity-0 group-hover/btn:opacity-100 transition-opacity duration-300"></div>
                  <span className="relative flex items-center space-x-3">
                    <Sparkles className="w-6 h-6" />
                    <span>Start AI Analysis</span>
                  </span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(51, 65, 85, 0.3);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: linear-gradient(to bottom, #6366f1, #a855f7);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(to bottom, #4f46e5, #9333ea);
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        .delay-1000 {
          animation-delay: 1s;
        }
        .delay-2000 {
          animation-delay: 2s;
        }
      `}</style>
    </div>
  );
};

export default Dashboard;