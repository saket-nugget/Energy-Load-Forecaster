
import { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Activity, Zap, AlertTriangle, Calendar } from 'lucide-react'
import clsx from 'clsx'

// Types
interface ForecastResponse {
  forecast: number[];
  unit: string;
  horizon: number;
}

interface Status {
  status: string;
  model_loaded: boolean;
  preprocessor_loaded: boolean;
}

function App() {
  const [forecast, setForecast] = useState<number[]>([])
  const [loading, setLoading] = useState(false)
  const [systemStatus, setSystemStatus] = useState<Status | null>(null)
  const [error, setError] = useState<string | null>(null)

  const API_URL = 'http://localhost:8000'

  useEffect(() => {
    checkHealth()
  }, [])

  const checkHealth = async () => {
    try {
      const res = await axios.get(`${API_URL}/health`)
      setSystemStatus(res.data)
    } catch (err) {
      console.error(err)
      setSystemStatus({ status: 'error', model_loaded: false, preprocessor_loaded: false })
    }
  }

  const getForecast = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await axios.get<ForecastResponse>(`${API_URL}/predict/demo`)
      setForecast(res.data.forecast)
    } catch (err) {
      setError('Failed to fetch forecast. Ensure backend is running.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // Mock data for chart visualization if no forecast yet, or display the forecast
  const chartData = forecast.length > 0
    ? forecast.map((val, idx) => ({ name: `Hour ${idx + 1}`, value: val }))
    : []

  return (
    <div className="min-h-screen w-full bg-slate-950 text-slate-50 font-sans selection:bg-indigo-500/30">

      {/* Navbar */}
      <nav className="border-b border-indigo-500/10 bg-slate-900/50 backdrop-blur-xl fixed w-full z-10 top-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-500/10 rounded-lg">
                <Zap className="w-6 h-6 text-indigo-400" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                ELF Dashboard
              </span>
            </div>
            <div className="flex items-center gap-4">
              <StatusBadge label="Backend" active={systemStatus?.status === 'ok'} />
              <StatusBadge label="Model" active={systemStatus?.model_loaded || false} />
            </div>
          </div>
        </div>
      </nav>

      <main className="pt-24 pb-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-8">

        {/* Hero / Actions */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

          {/* Controls Card */}
          <div className="lg:col-span-1 space-y-6">
            <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm shadow-xl hover:shadow-indigo-500/5 transition-all">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-indigo-400" />
                Control Center
              </h2>
              <p className="text-slate-400 text-sm mb-6">
                Trigger a new short-term load forecast based on the latest available data.
              </p>

              <button
                onClick={getForecast}
                disabled={loading || !systemStatus?.model_loaded}
                className={clsx(
                  "w-full py-3 px-4 rounded-xl font-medium transition-all duration-200 flex items-center justify-center gap-2",
                  loading || !systemStatus?.model_loaded
                    ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                    : "bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20 active:scale-[0.98]"
                )}
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Running Model...
                  </>
                ) : (
                  <>Run Forecast</>
                )}
              </button>

              {error && (
                <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                  {error}
                </div>
              )}
            </div>

            {/* Stats Card (Mockup for now) */}
            <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm shadow-xl">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Calendar className="w-5 h-5 text-cyan-400" />
                System Info
              </h2>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between py-2 border-b border-slate-800">
                  <span className="text-slate-400">Model Type</span>
                  <span className="text-slate-200">Transformer (TSFM)</span>
                </div>
                <div className="flex justify-between py-2 border-b border-slate-800">
                  <span className="text-slate-400">Horizon</span>
                  <span className="text-slate-200">24 Hours</span>
                </div>
                <div className="flex justify-between py-2 border-b border-slate-800">
                  <span className="text-slate-400">Input Length</span>
                  <span className="text-slate-200">168 Hours</span>
                </div>
              </div>
            </div>
          </div>

          {/* Visualization Card */}
          <div className="lg:col-span-2 p-6 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-sm shadow-xl min-h-[400px]">
            <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-400" />
              Forecast Visualization
            </h2>

            {forecast.length > 0 ? (
              <div className="h-[350px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                    <XAxis
                      dataKey="name"
                      stroke="#94a3b8"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      stroke="#94a3b8"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(value) => `${value} MW`}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                      itemStyle={{ color: '#e2e8f0' }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="value"
                      name="Forecasted Load (MW)"
                      stroke="#818cf8"
                      strokeWidth={3}
                      dot={{ r: 4, fill: '#818cf8', strokeWidth: 0 }}
                      activeDot={{ r: 6, fill: '#a5b4fc' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-slate-500 space-y-4">
                <Activity className="w-12 h-12 opacity-20" />
                <p>Run a forecast to see the load prediction.</p>
              </div>
            )}
          </div>

        </div>

      </main>
    </div>
  )
}

function StatusBadge({ label, active }: { label: string; active: boolean }) {
  return (
    <div className={clsx(
      "flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors",
      active
        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
        : "bg-red-500/10 text-red-400 border-red-500/20"
    )}>
      <div className={clsx("w-1.5 h-1.5 rounded-full", active ? "bg-emerald-400" : "bg-red-400")} />
      {label}
    </div>
  )
}

export default App
