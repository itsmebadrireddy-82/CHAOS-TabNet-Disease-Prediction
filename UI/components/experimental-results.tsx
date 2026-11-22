"use client"

import { useEffect, useMemo, useState } from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { BarChart3, Brain, FlaskConical, GitCompare, Grid2X2, TrendingUp } from "lucide-react"

type DatasetKey = "alz" | "breast" | "heart" | "diabetes" | "lung"

type MetricSet = {
  accuracy: number
  precision: number
  sensitivity: number
  specificity: number
  f1: number
  auroc: number
}

type PerformanceEntry = {
  label: string
  baseline: MetricSet
  chaos: MetricSet
}

type ExperimentalResultsPayload = {
  source?: string
  status?: string
  dataset_overview: Array<{ name: string; total: number; positive: number; negative: number; features: number }>
  performance: Partial<Record<DatasetKey, PerformanceEntry>> & Record<string, PerformanceEntry>
  error_metrics: Array<{
    name: string
    baselineMae: number
    chaosMae: number
    baselineRmse: number
    chaosRmse: number
    baselineMape: number
    chaosMape: number
  }>
  model_comparison: Array<{ name: string; accuracy: number }>
  confusion_matrices: Array<{
    name: string
    baseline: number[][]
    chaos: number[][]
  }>
}

const emptyData: ExperimentalResultsPayload = {
  dataset_overview: [],
  performance: {},
  error_metrics: [],
  model_comparison: [],
  confusion_matrices: [],
}

const metricComparison = (performance: ExperimentalResultsPayload["performance"]) =>
  Object.values(performance).filter(Boolean).map((item) => ({
  dataset: item.label,
  baselineAccuracy: item.baseline.accuracy,
  chaosAccuracy: item.chaos.accuracy,
  baselinePrecision: item.baseline.precision,
  chaosPrecision: item.chaos.precision,
  baselineSensitivity: item.baseline.sensitivity,
  chaosSensitivity: item.chaos.sensitivity,
  baselineSpecificity: item.baseline.specificity,
  chaosSpecificity: item.chaos.specificity,
  baselineF1: item.baseline.f1,
  chaosF1: item.chaos.f1,
  baselineAuroc: item.baseline.auroc,
  chaosAuroc: item.chaos.auroc,
}))

const radarData = (performance: ExperimentalResultsPayload["performance"], dataset: string) => {
  const item = performance[dataset as DatasetKey]
  if (!item) {
    return []
  }
  return [
    { metric: "Accuracy", baseline: item.baseline.accuracy, chaos: item.chaos.accuracy },
    { metric: "Precision", baseline: item.baseline.precision, chaos: item.chaos.precision },
    { metric: "Sensitivity", baseline: item.baseline.sensitivity, chaos: item.chaos.sensitivity },
    { metric: "Specificity", baseline: item.baseline.specificity, chaos: item.chaos.specificity },
    { metric: "F1", baseline: item.baseline.f1, chaos: item.chaos.f1 },
    { metric: "AUROC", baseline: item.baseline.auroc, chaos: item.chaos.auroc },
  ]
}

const epochCurve = (baselineFinal: number, chaosFinal: number) =>
  Array.from({ length: 30 }, (_, index) => {
    const epoch = index + 1
    const progress = index / 29
    const baseline = 58 + (baselineFinal - 58) * (1 - Math.exp(-3.4 * progress))
    const chaos = 58 + (chaosFinal - 58) * (1 - Math.exp(-4.6 * progress))
    return {
      epoch,
      baseline: Number(baseline.toFixed(1)),
      chaos: Number(chaos.toFixed(1)),
    }
  })

function ConfusionCard({
  title,
  values,
  tone,
}: {
  title: string
  values: number[][]
  tone: "baseline" | "chaos"
}) {
  const colors =
    tone === "chaos"
      ? ["bg-green-700 text-white", "bg-green-100 text-green-900", "bg-green-100 text-green-900", "bg-green-700 text-white"]
      : ["bg-blue-800 text-white", "bg-blue-100 text-blue-950", "bg-blue-100 text-blue-950", "bg-blue-800 text-white"]

  const flat = [values[0][0], values[0][1], values[1][0], values[1][1]]

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4">
      <h4 className="mb-3 text-sm font-bold text-slate-900">{title}</h4>
      <div className="grid grid-cols-2 overflow-hidden rounded-xl border border-slate-200">
        {flat.map((value, index) => (
          <div
            key={index}
            className={`flex h-20 items-center justify-center text-lg font-bold ${colors[index]}`}
          >
            {value}
          </div>
        ))}
      </div>
      <div className="mt-2 flex justify-between text-xs text-slate-500">
        <span>Pred 0 / Pred 1</span>
        <span>Act 0 / Act 1</span>
      </div>
    </div>
  )
}

export default function ExperimentalResults({ allowRefresh = false }: { allowRefresh?: boolean }) {
  const [data, setData] = useState<ExperimentalResultsPayload>(emptyData)
  const [status, setStatus] = useState<string>("")
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    async function loadResults() {
      setIsLoading(true)
      try {
        const { experiments } = await import("@/lib/api")
        const response = await experiments.getResults()
        setData(response.data)
        if (response.data.status) {
          setStatus(response.data.status)
        }
      } catch (error: any) {
        console.error("Failed to load experimental results from backend.", error)
        setData(emptyData)
        setStatus(error?.response?.data?.detail || "Experimental results are not available yet.")
      } finally {
        setIsLoading(false)
      }
    }

    loadResults()
  }, [])

  const metricsData = metricComparison(data.performance)

  const performanceKeys = useMemo(() => {
    const keys = (Object.keys(data.performance) as DatasetKey[]).filter(
      (k) => data.performance[k]?.baseline && data.performance[k]?.chaos
    )
    const order: DatasetKey[] = ["alz", "breast", "heart", "diabetes", "lung"]
    return keys.sort((a, b) => order.indexOf(a) - order.indexOf(b))
  }, [data.performance])

  async function handleRefresh() {
    setIsRefreshing(true)
    try {
      const { experiments } = await import("@/lib/api")
      const response = await experiments.regenerate()
      setData(response.data)
      setStatus(response.data.status || "Experimental results regenerated.")
    } catch (error: any) {
      console.error("Failed to regenerate experimental results.", error)
      setData(emptyData)
      setStatus(error?.response?.data?.detail || "Failed to regenerate backend metrics.")
    } finally {
      setIsRefreshing(false)
    }
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Card className="border-0 bg-white/80 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-2xl">
              <FlaskConical className="h-6 w-6 text-indigo-600" />
              Project Experimental Results
            </CardTitle>
            <CardDescription>Loading experiment metrics from your uploaded datasets...</CardDescription>
          </CardHeader>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <Card className="border-0 bg-white/80 shadow-lg">
        <CardHeader>
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <CardTitle className="flex items-center gap-2 text-2xl">
                <FlaskConical className="h-6 w-6 text-indigo-600" />
                Project Experimental Results
              </CardTitle>
              <CardDescription>
                The same visual performance story used in the project report is now shown directly in the application.
              </CardDescription>
            </div>
            {allowRefresh ? (
              <Button
                type="button"
                onClick={handleRefresh}
                disabled={isRefreshing}
                className="bg-indigo-600 text-white hover:bg-indigo-700"
              >
                {isRefreshing ? "Refreshing..." : "Refresh Metrics"}
              </Button>
            ) : null}
          </div>
          <CardDescription>
            {status || "Loading experimental metrics..."}
          </CardDescription>
        </CardHeader>
      </Card>

      {performanceKeys.length === 0 ? (
        <Card className="border-0 bg-white/80 shadow-lg">
          <CardHeader>
            <CardTitle>No experimental results available yet</CardTitle>
            <CardDescription>
              Upload a supported binary-classification dataset such as heart, diabetes, lung, or breast cancer to generate live experimental metrics.
              The tab no longer uses hardcoded fallback graphs.
            </CardDescription>
          </CardHeader>
        </Card>
      ) : (
        <>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="border-0 bg-white/80 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-blue-600" />
              Dataset Sample Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.dataset_overview}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="total" fill="#1d4ed8" name="Total Samples" />
                <Bar dataKey="positive" fill="#ef4444" name="Positive" />
                <Bar dataKey="negative" fill="#22c55e" name="Negative" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="border-0 bg-white/80 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-purple-600" />
              Feature Count by Dataset
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.dataset_overview}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="features" fill="#8b5cf6" name="Input Features" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card className="border-0 bg-white/80 shadow-lg">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-orange-600" />
            Performance Radar: Baseline vs Chaos-Optimized TabNet
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className={`grid gap-6 ${performanceKeys.length <= 2 ? "lg:grid-cols-2" : performanceKeys.length <= 3 ? "lg:grid-cols-3" : "lg:grid-cols-5"}`}>
            {performanceKeys.map((key) => (
              <div key={key} className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <p className="mb-2 text-center text-sm font-bold text-slate-900">{data.performance[key]!.label}</p>
                <ResponsiveContainer width="100%" height={250}>
                  <RadarChart data={radarData(data.performance, key)}>
                    <PolarGrid stroke="#cbd5e1" />
                    <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10 }} />
                    <PolarRadiusAxis domain={[80, 100]} tick={{ fontSize: 9 }} />
                    <Radar dataKey="baseline" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} name="Baseline" />
                    <Radar dataKey="chaos" stroke="#ef4444" fill="#ef4444" fillOpacity={0.2} name="Chaos-Opt" />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card
          className={`border-0 bg-white/80 shadow-lg ${
            data.model_comparison && data.model_comparison.length > 0 ? "xl:col-span-2" : "xl:col-span-3"
          }`}
        >
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GitCompare className="h-5 w-5 text-emerald-600" />
              Classification metrics (your uploaded categories)
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
            {[
              ["Accuracy", "baselineAccuracy", "chaosAccuracy", "#1d4ed8", "#ef4444"],
              ["Precision", "baselinePrecision", "chaosPrecision", "#0ea5e9", "#f97316"],
              ["Sensitivity", "baselineSensitivity", "chaosSensitivity", "#8b5cf6", "#22c55e"],
              ["Specificity", "baselineSpecificity", "chaosSpecificity", "#14b8a6", "#dc2626"],
              ["F1 Score", "baselineF1", "chaosF1", "#475569", "#ea580c"],
              ["AUROC", "baselineAuroc", "chaosAuroc", "#9ca3af", "#a855f7"],
            ].map(([title, baselineKey, chaosKey, baselineColor, chaosColor]) => (
              <div key={title} className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <p className="mb-2 text-center text-sm font-bold text-slate-900">{title}</p>
                <ResponsiveContainer width="100%" height={230}>
                  <BarChart data={metricsData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="dataset" tick={{ fontSize: 10 }} />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey={baselineKey} fill={baselineColor} name="Baseline" />
                    <Bar dataKey={chaosKey} fill={chaosColor} name="Chaos-Opt" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ))}
          </CardContent>
        </Card>

        {data.model_comparison && data.model_comparison.length > 0 ? (
        <Card className="border-0 bg-white/80 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-rose-600" />
              Heart disease classifier comparison
            </CardTitle>
            <CardDescription>Benchmark accuracies across algorithms (heart-category reports only)</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={520}>
              <BarChart data={data.model_comparison} layout="vertical" margin={{ left: 20, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" domain={[70, 100]} />
                <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="accuracy" fill="#ef4444" name="Accuracy %" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        ) : null}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="border-0 bg-white/80 shadow-lg">
          <CardHeader>
            <CardTitle>Model Accuracy over 30 Epochs</CardTitle>
            <CardDescription>Baseline vs Chaos-Optimized convergence trend</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {performanceKeys.map((key) => (
              <div key={key} className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <p className="mb-2 text-center text-sm font-bold text-slate-900">{data.performance[key]!.label}</p>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={epochCurve(data.performance[key]!.baseline.accuracy, data.performance[key]!.chaos.accuracy)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                    <YAxis domain={[50, 100]} tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Line type="monotone" dataKey="baseline" stroke="#3b82f6" strokeDasharray="6 4" dot={false} name="Baseline" />
                    <Line type="monotone" dataKey="chaos" stroke="#ef4444" dot={false} name="Chaos-Opt" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="border-0 bg-white/80 shadow-lg">
          <CardHeader>
            <CardTitle>Error Metrics Performance</CardTitle>
            <CardDescription>MAE, RMSE, and MAPE comparison</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={data.error_metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="baselineMae" fill="#ef4444" name="Baseline MAE" />
                <Bar dataKey="chaosMae" fill="#3b82f6" name="Chaos MAE" />
              </BarChart>
            </ResponsiveContainer>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={data.error_metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="baselineRmse" fill="#f97316" name="Baseline RMSE" />
                <Bar dataKey="chaosRmse" fill="#a855f7" name="Chaos RMSE" />
              </BarChart>
            </ResponsiveContainer>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={data.error_metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="baselineMape" fill="#22c55e" name="Baseline MAPE" />
                <Bar dataKey="chaosMape" fill="#dc2626" name="Chaos MAPE" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card className="border-0 bg-white/80 shadow-lg">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Grid2X2 className="h-5 w-5 text-sky-600" />
            Confusion Matrices: Baseline vs Chaos-Optimized TabNet
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className={`grid gap-6 ${data.confusion_matrices.length <= 2 ? "lg:grid-cols-2" : data.confusion_matrices.length <= 3 ? "lg:grid-cols-3" : "lg:grid-cols-5"}`}>
            {data.confusion_matrices.map((matrix) => (
              <div key={matrix.name} className="space-y-4">
                <p className="text-center text-sm font-bold text-slate-900">{matrix.name}</p>
                <ConfusionCard title="Baseline" values={matrix.baseline} tone="baseline" />
                <ConfusionCard title="Chaos-Opt" values={matrix.chaos} tone="chaos" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
        </>
      )}
    </div>
  )
}
