"use client"

import { useCallback, useEffect, useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Activity,
  BarChart3,
  Brain,
  ClipboardList,
  Database,
  FlaskConical,
  TrendingUp,
  Upload,
  Zap,
} from "lucide-react"
import WorkspaceHeader from "./workspace-header"
import CSVUpload from "./csv-upload"
import DataPreprocessing from "./data-preprocessing"
import ChatInterface from "./chat-interface"
import PredictionResults from "./prediction-results"
import HealthAnalytics from "./health-analytics"
import ExperimentalResults from "./experimental-results"

interface WorkspaceDashboardProps {
  userName: string
  onLogout: () => void
}

type WorkspaceStats = {
  total_datasets: number
  total_records: number
  active_models: number
  accuracy_rate: number
  has_uploaded_data: boolean
  diseases: { disease_type: string; dataset_count: number; record_count: number }[]
  analytics_models: { name: string; accuracy: number }[]
}

type UserStats = {
  total_assessments: number
  health_score: number
  risk_factors: number
}

export default function WorkspaceDashboard({ userName, onLogout }: WorkspaceDashboardProps) {
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([])
  const [predictions, setPredictions] = useState<any[]>([])
  const [activeTab, setActiveTab] = useState("upload")
  const [workspaceStats, setWorkspaceStats] = useState<WorkspaceStats>({
    total_datasets: 0,
    total_records: 0,
    active_models: 0,
    accuracy_rate: 0,
    has_uploaded_data: false,
    diseases: [],
    analytics_models: [],
  })
  const [userStats, setUserStats] = useState<UserStats>({
    total_assessments: 0,
    health_score: 0,
    risk_factors: 0,
  })

  const loadWorkspaceData = useCallback(async () => {
    const { dashboard } = await import("@/lib/api")

    try {
      const [workspaceRes, uploadsRes, userRes] = await Promise.all([
        dashboard.getWorkspaceStats(),
        dashboard.listWorkspaceUploads(),
        dashboard.getUserStats(),
      ])

      setWorkspaceStats(workspaceRes.data)
      setUploadedFiles(uploadsRes.data.filenames ?? [])
      setUserStats(userRes.data)
    } catch (error) {
      console.error("Failed to fetch workspace data", error)
    }
  }, [])

  useEffect(() => {
    loadWorkspaceData()
  }, [loadWorkspaceData])

  const handleFileUpload = (fileName: string) => {
    setUploadedFiles((prev) => [...prev, fileName])
  }

  const handlePrediction = (data: any) => {
    setPredictions((prev) => [data, ...prev])
    setActiveTab("results")
    loadWorkspaceData()
  }

  const hasAnalyticsData = workspaceStats.has_uploaded_data || workspaceStats.total_datasets > 0

  const summaryCards = [
    {
      title: "Total Datasets",
      value: workspaceStats.total_datasets.toString(),
      helper: hasAnalyticsData ? "Ready for disease-specific workflows" : "Upload your first CSV",
      icon: Database,
      color: "from-blue-500 to-cyan-500",
    },
    {
      title: "Processed Records",
      value: workspaceStats.total_records.toString(),
      helper: hasAnalyticsData ? "Stored in your workspace" : "No uploaded records yet",
      icon: Zap,
      color: "from-violet-500 to-fuchsia-500",
    },
    {
      title: "Assessments",
      value: userStats.total_assessments.toString(),
      helper: userStats.total_assessments > 0 ? "Saved prediction history" : "Run a prediction to start tracking",
      icon: ClipboardList,
      color: "from-orange-500 to-red-500",
    },
    {
      title: "Reported Accuracy",
      value: hasAnalyticsData ? `${workspaceStats.accuracy_rate}%` : "—",
      helper: hasAnalyticsData ? "Benchmark view from experiments" : "Available after upload",
      icon: Brain,
      color: "from-emerald-500 to-green-500",
    },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-fuchsia-50">
      <WorkspaceHeader userName={userName} onLogout={onLogout} />

      <main className="container mx-auto p-6 lg:p-8">
        <div className="mb-10">
          <div className="mb-4 flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-fuchsia-600 shadow-lg shadow-blue-500/30">
              <Brain className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="bg-gradient-to-r from-slate-900 to-slate-600 bg-clip-text text-4xl font-bold text-transparent">
                Unified Disease Intelligence Workspace
              </h1>
              <p className="mt-1 text-lg text-slate-600">
                Manage datasets, inspect experiments, and generate disease risk predictions from one shared flow.
              </p>
            </div>
          </div>
        </div>

        <div className="mb-8 grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
          {summaryCards.map((card, idx) => (
            <div key={card.title} className="group relative animate-slide-up" style={{ animationDelay: `${idx * 0.08}s` }}>
              <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${card.color} opacity-10 blur-xl transition-all duration-500 group-hover:opacity-20 group-hover:blur-2xl`} />
              <Card className="relative overflow-hidden border-0 bg-white/85 shadow-lg backdrop-blur-sm transition-all duration-300 hover:shadow-xl">
                <CardContent className="p-6">
                  <div className="mb-4 flex items-start justify-between">
                    <div className={`flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br ${card.color} shadow-md`}>
                      <card.icon className="h-6 w-6 text-white" />
                    </div>
                  </div>
                  <p className="mb-2 text-sm font-medium text-slate-600">{card.title}</p>
                  <p className="text-3xl font-bold text-slate-900">{card.value}</p>
                  <p className="mt-2 text-sm text-slate-500">{card.helper}</p>
                </CardContent>
              </Card>
            </div>
          ))}
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="h-auto rounded-2xl border border-slate-200 bg-white/85 p-1.5 shadow-lg backdrop-blur-sm flex flex-wrap justify-start">
            <TabsTrigger value="upload" className="rounded-xl px-5 py-3 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-500 data-[state=active]:to-fuchsia-600 data-[state=active]:text-white">
              <Upload className="mr-2 h-5 w-5" />
              Upload
            </TabsTrigger>
            <TabsTrigger value="processing" className="rounded-xl px-5 py-3 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-500 data-[state=active]:to-fuchsia-600 data-[state=active]:text-white">
              <BarChart3 className="mr-2 h-5 w-5" />
              Processing
            </TabsTrigger>
            <TabsTrigger value="prediction" className="rounded-xl px-5 py-3 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-500 data-[state=active]:to-fuchsia-600 data-[state=active]:text-white">
              <Activity className="mr-2 h-5 w-5" />
              Prediction
            </TabsTrigger>
            <TabsTrigger value="results" className="rounded-xl px-5 py-3 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-500 data-[state=active]:to-fuchsia-600 data-[state=active]:text-white">
              <TrendingUp className="mr-2 h-5 w-5" />
              Results
            </TabsTrigger>
            <TabsTrigger value="analytics" className="rounded-xl px-5 py-3 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-500 data-[state=active]:to-fuchsia-600 data-[state=active]:text-white">
              <Zap className="mr-2 h-5 w-5" />
              Analytics
            </TabsTrigger>
            <TabsTrigger value="experiments" className="rounded-xl px-5 py-3 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-500 data-[state=active]:to-fuchsia-600 data-[state=active]:text-white">
              <FlaskConical className="mr-2 h-5 w-5" />
              Experiments
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-6">
            <CSVUpload onFileUpload={handleFileUpload} onUploadComplete={loadWorkspaceData} />

            {uploadedFiles.length > 0 ? (
              <Card className="border-0 bg-white/85 shadow-lg backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-3 text-2xl">
                    <Database className="h-6 w-6 text-blue-600" />
                    Uploaded Datasets
                  </CardTitle>
                  <CardDescription>
                    These files are available for chat prompts, disease selection, and experimental filtering in your workspace.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {uploadedFiles.map((file, idx) => (
                      <div
                        key={`${file}-${idx}`}
                        className="flex items-center justify-between rounded-xl border border-slate-200 bg-gradient-to-r from-slate-50 to-blue-50 p-4"
                      >
                        <div className="flex items-center gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-fuchsia-600">
                            <Database className="h-5 w-5 text-white" />
                          </div>
                          <span className="text-base font-semibold text-slate-900">{file}</span>
                        </div>
                        <span className="rounded-full bg-green-100 px-3 py-1 text-sm font-medium text-green-700">Ready</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ) : null}
          </TabsContent>

          <TabsContent value="processing" className="space-y-6">
            <DataPreprocessing uploadedFiles={uploadedFiles} onPreprocess={() => undefined} />
          </TabsContent>

          <TabsContent value="prediction">
            <ChatInterface onPredictionComplete={handlePrediction} />
          </TabsContent>

          <TabsContent value="results">
            <PredictionResults predictions={predictions} />
          </TabsContent>

          <TabsContent value="analytics">
            <HealthAnalytics predictions={predictions} />
          </TabsContent>

          <TabsContent value="experiments" className="space-y-6">
            {!hasAnalyticsData ? (
              <Card className="border-0 bg-white/85 shadow-lg backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-2xl">Experiments unlock after your first upload</CardTitle>
                  <CardDescription>
                    Upload at least one disease dataset in this workspace to view filtered experimental metrics and refresh the benchmark payload.
                  </CardDescription>
                </CardHeader>
              </Card>
            ) : null}
            <ExperimentalResults allowRefresh />
          </TabsContent>
        </Tabs>
      </main>

      <style jsx>{`
        @keyframes slide-up {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-slide-up {
          animation: slide-up 0.6s ease-out forwards;
          opacity: 0;
        }
      `}</style>
    </div>
  )
}
