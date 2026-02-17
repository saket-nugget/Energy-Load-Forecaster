import { useState, useEffect } from 'react'
import axios from 'axios'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, ComposedChart, Scatter } from 'recharts'
import { Activity, Zap, AlertTriangle, Calendar, RefreshCw, Layers, Clock, CloudRain, Wind, Terminal, Sun, Droplets, Download } from 'lucide-react'
import clsx from 'clsx'
import { motion, AnimatePresence } from 'framer-motion'
import { format, addHours, getHours } from 'date-fns'

// Types
interface ForecastResponse {
    forecast: number[];
    anomalies: boolean[];
    unit: string;
    horizon_hours: number;
    days_requested: number;
}

interface Status {
    status: string;
    model_loaded: boolean;
    preprocessor_loaded: boolean;
}

interface LogEntry {
    timestamp: Date;
    message: string;
    type: 'info' | 'success' | 'warning' | 'error';
}

function App() {
    const [forecast, setForecast] = useState<number[]>([])
    const [anomalies, setAnomalies] = useState<boolean[]>([])
    const [loading, setLoading] = useState(false)
    const [systemStatus, setSystemStatus] = useState<Status | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [horizonDays, setHorizonDays] = useState<number>(1)
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
    const [logs, setLogs] = useState<LogEntry[]>([])

    const API_URL = 'http://localhost:8000'

    const addLog = (message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') => {
        setLogs(prev => [{ timestamp: new Date(), message, type }, ...prev].slice(0, 50))
    }

    useEffect(() => {
        checkHealth()
        addLog("System initialized. Waiting for user input...", 'info')
        const interval = setInterval(checkHealth, 30000) // Poll every 30s
        return () => clearInterval(interval)
    }, [])

    const checkHealth = async () => {
        try {
            const res = await axios.get(`${API_URL}/health`)
            setSystemStatus(res.data)
            if (res.data.status === 'ok' && !systemStatus?.model_loaded) {
                addLog("Connected to Backend API.", 'success')
            }
        } catch (err) {
            console.error(err)
            setSystemStatus({ status: 'error', model_loaded: false, preprocessor_loaded: false })
            addLog("Connection lost to Backend API.", 'warning')
        }
    }

    const getForecast = async () => {
        setLoading(true)
        setError(null)
        setForecast([])
        setAnomalies([])
        addLog(`Requesting ${horizonDays}-day forecast from Transformer model...`, 'info')

        try {
            const res = await axios.get<ForecastResponse>(`${API_URL}/predict/demo?days=${horizonDays}`)

            // Simulate a small delay for the "premium feel" if response is too instant
            if (res.data.forecast.length < 100) {
                await new Promise(r => setTimeout(r, 800));
            }

            setForecast(res.data.forecast)
            setAnomalies(res.data.anomalies || [])
            setLastUpdated(new Date())

            const anomalyCount = (res.data.anomalies || []).filter(a => a).length
            if (anomalyCount > 0) {
                addLog(`Analysis complete. Found ${anomalyCount} potential anomalies!`, 'warning')
            } else {
                addLog(`Received ${res.data.forecast.length} data points. Rendering visualization.`, 'success')
            }
        } catch (err) {
            setError('Failed to fetch forecast. Ensure backend is running.')
            addLog("Inference failed. Check console for details.", 'error')
            console.error(err)
        } finally {
            setLoading(false)
        }
    }

    const exportData = () => {
        if (forecast.length === 0) return

        const headers = ["Timestamp", "Load (MW)", "Anomaly"]
        const rows = chartData.map(d => [d.fullDate, d.value, d.isAnomaly ? "YES" : "NO"])

        const csvContent = [
            headers.join(","),
            ...rows.map(r => r.join(","))
        ].join("\n")

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
        const url = URL.createObjectURL(blob)
        const link = document.createElement("a")
        link.setAttribute("href", url)
        link.setAttribute("download", `forecast_${format(new Date(), 'yyyyMMdd_HHmm')}.csv`)
        link.style.visibility = 'hidden'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        addLog("Forecast exported to CSV.", 'success')
    }

    // Generate chart data with timestamps
    const chartData = forecast.length > 0
        ? forecast.map((val, idx) => {
            // Start from "Next Hour" (mock logic: current time + idx + 1 hours)
            const date = addHours(new Date(), idx + 1)
            return {
                date: date.toISOString(),
                label: horizonDays <= 1 ? format(date, 'HH:mm') : format(date, 'dd MMM HH:mm'),
                value: Math.round(val),
                fullDate: format(date, 'PPpp'),
                hour: getHours(date),
                isAnomaly: anomalies[idx] || false,
                anomalyValue: (anomalies[idx]) ? Math.round(val) : null // For Scatter plot
            }
        })
        : []

    // Load Breakdown Logic
    const breakdownData = forecast.length > 0 ? [
        { name: 'Morning (6-12)', value: Math.round(chartData.filter(d => d.hour >= 6 && d.hour < 12).reduce((acc, curr) => acc + curr.value, 0) / (chartData.filter(d => d.hour >= 6 && d.hour < 12).length || 1)), color: '#fbbf24' },
        { name: 'Afternoon (12-18)', value: Math.round(chartData.filter(d => d.hour >= 12 && d.hour < 18).reduce((acc, curr) => acc + curr.value, 0) / (chartData.filter(d => d.hour >= 12 && d.hour < 18).length || 1)), color: '#f97316' },
        { name: 'Evening (18-24)', value: Math.round(chartData.filter(d => d.hour >= 18 || d.hour < 6).reduce((acc, curr) => acc + curr.value, 0) / (chartData.filter(d => d.hour >= 18 || d.hour < 6).length || 1)), color: '#818cf8' },
    ] : []

    const stats = forecast.length > 0 ? {
        max: Math.max(...forecast).toLocaleString(),
        min: Math.min(...forecast).toLocaleString(),
        avg: Math.round(forecast.reduce((a, b) => a + b, 0) / forecast.length).toLocaleString()
    } : null

    return (
        <div className="min-h-screen w-full text-slate-100 font-sans selection:bg-indigo-500/30 overflow-x-hidden">

            {/* Navbar */}
            <nav className="glass-panel fixed w-full z-50 top-0">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between h-20">
                        <div className="flex items-center gap-4">
                            <div className="p-2.5 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-xl border border-indigo-500/20 shadow-lg shadow-indigo-500/10">
                                <Zap className="w-6 h-6 text-indigo-400" />
                            </div>
                            <div>
                                <h1 className="text-xl font-bold bg-gradient-to-r from-white via-indigo-100 to-indigo-200 bg-clip-text text-transparent">
                                    ELF Dashboard
                                </h1>
                                <p className="text-xs text-slate-400 font-medium tracking-wide">AI ENERGY FORECASTER</p>
                            </div>
                        </div>

                        <div className="flex items-center gap-6">
                            <StatusIndicator
                                label="Backend AI"
                                active={systemStatus?.status === 'ok'}
                                loading={!systemStatus}
                            />
                            <StatusIndicator
                                label="Model Ready"
                                active={systemStatus?.model_loaded || false}
                                loading={!systemStatus}
                            />
                        </div>
                    </div>
                </div>
            </nav>

            <main className="pt-32 pb-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-8">

                {/* Header Section */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col md:flex-row md:items-end justify-between gap-6"
                >
                    <div>
                        <h2 className="text-3xl font-bold text-white mb-2">Load Forecast</h2>
                        <p className="text-slate-400 max-w-xl text-lg">
                            Predictive energy consumption models powered by Transformer architecture.
                        </p>
                    </div>

                    <div className="flex items-center gap-4">
                        {/* Horizon Selector */}
                        <div className="flex items-center gap-2 bg-slate-900/50 p-1.5 rounded-xl border border-slate-700/50">
                            {[1, 3, 7, 30].map((days) => (
                                <button
                                    key={days}
                                    onClick={() => setHorizonDays(days)}
                                    className={clsx(
                                        "px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                                        horizonDays === days
                                            ? "bg-indigo-500/20 text-indigo-300 shadow-lg shadow-indigo-500/10 border border-indigo-500/20"
                                            : "text-slate-400 hover:text-slate-200 hover:bg-white/5"
                                    )}
                                >
                                    {days} Day{days > 1 ? 's' : ''}
                                </button>
                            ))}
                        </div>

                        {/* Export Button */}
                        <button
                            onClick={exportData}
                            disabled={forecast.length === 0}
                            className="p-3 rounded-xl bg-slate-800 border border-slate-700 text-slate-400 hover:text-white hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            title="Export to CSV"
                        >
                            <Download className="w-5 h-5" />
                        </button>
                    </div>
                </motion.div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                    {/* Main Chart Card */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.1 }}
                        className="lg:col-span-2 glass-card rounded-3xl p-1 overflow-hidden relative group"
                    >
                        {/* Glow Effect */}
                        <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl -z-10 group-hover:bg-indigo-500/15 transition-colors duration-500" />

                        <div className="bg-slate-900/40 rounded-[20px] p-6 h-[450px] flex flex-col">
                            <div className="flex justify-between items-start mb-6">
                                <div>
                                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                        <Activity className="w-5 h-5 text-indigo-400" />
                                        Real-time Visualization
                                    </h3>
                                    {lastUpdated && (
                                        <p className="text-xs text-slate-500 mt-1 flex items-center gap-1.5">
                                            <Clock className="w-3 h-3" />
                                            Updated {format(lastUpdated, 'HH:mm:ss')}
                                        </p>
                                    )}
                                </div>

                                {/* Run Button */}
                                <button
                                    onClick={getForecast}
                                    disabled={loading || !systemStatus?.model_loaded}
                                    className={clsx(
                                        "relative overflow-hidden group/btn px-6 py-2.5 rounded-xl font-semibold text-sm transition-all duration-300 flex items-center gap-2",
                                        loading || !systemStatus?.model_loaded
                                            ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                                            : "bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/40 active:scale-[0.98]"
                                    )}
                                >
                                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover/btn:translate-x-[100%] transition-transform duration-700" />
                                    {loading ? (
                                        <>
                                            <RefreshCw className="w-4 h-4 animate-spin" />
                                            Running Model...
                                        </>
                                    ) : (
                                        <>
                                            <Zap className="w-4 h-4 fill-current" />
                                            Generate Forecast
                                        </>
                                    )}
                                </button>
                            </div>

                            {error && (
                                <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm flex items-center gap-3"
                                >
                                    <AlertTriangle className="w-5 h-5 shrink-0" />
                                    {error}
                                </motion.div>
                            )}

                            <div className="flex-1 w-full min-h-0">
                                {chartData.length > 0 ? (
                                    <ResponsiveContainer width="100%" height="100%">
                                        <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                                            <defs>
                                                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4} />
                                                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} vertical={false} />
                                            <XAxis
                                                dataKey="label"
                                                stroke="#94a3b8"
                                                fontSize={11}
                                                tickLine={false}
                                                axisLine={false}
                                                tickMargin={10}
                                                minTickGap={30}
                                            />
                                            <YAxis
                                                stroke="#94a3b8"
                                                fontSize={11}
                                                tickLine={false}
                                                axisLine={false}
                                                tickFormatter={(value) => `${value / 1000}k`}
                                            />
                                            <Tooltip content={<CustomTooltip />} />
                                            <Area
                                                type="monotone"
                                                dataKey="value"
                                                stroke="#818cf8"
                                                strokeWidth={3}
                                                fillOpacity={1}
                                                fill="url(#colorValue)"
                                                animationDuration={1500}
                                            />
                                            {/* Anomaly Scatter Plot */}
                                            <Scatter
                                                dataKey="anomalyValue"
                                                fill="#ef4444"
                                                shape="circle"
                                                r={4}
                                                animationDuration={1000}
                                            />
                                        </ComposedChart>
                                    </ResponsiveContainer>
                                ) : (
                                    <div className="h-full flex flex-col items-center justify-center text-slate-500 space-y-4 border-2 border-dashed border-slate-700/50 rounded-2xl">
                                        <div className="p-4 bg-slate-800/50 rounded-full">
                                            <Activity className="w-8 h-8 opacity-40" />
                                        </div>
                                        <p className="text-sm font-medium">Model Ready. Start a forecast.</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </motion.div>

                    {/* Side Stats Card */}
                    <div className="lg:col-span-1 space-y-6">
                        <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.2 }}
                            className="glass-card rounded-3xl p-6"
                        >
                            <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                                <Layers className="w-5 h-5 text-purple-400" />
                                prediction Metrics
                            </h3>

                            <AnimatePresence mode='wait'>
                                {stats ? (
                                    <motion.div
                                        key="stats"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                        className="space-y-4"
                                    >
                                        <StatRow label="Peak Load" value={`${stats.max} MW`} color="text-emerald-400" />
                                        <StatRow label="Min Load" value={`${stats.min} MW`} color="text-blue-400" />
                                        <StatRow label="Average" value={`${stats.avg} MW`} color="text-indigo-400" />
                                        <div className="pt-4 border-t border-slate-700/50">
                                            <div className="flex justify-between items-center text-sm">
                                                <span className="text-slate-400">Confidence</span>
                                                <span className="text-emerald-400 font-semibold">95.2%</span>
                                            </div>
                                            <div className="w-full bg-slate-700/50 h-1.5 rounded-full mt-2 overflow-hidden">
                                                <div className="bg-emerald-500 h-full rounded-full w-[95%]" />
                                            </div>
                                        </div>
                                    </motion.div>
                                ) : (
                                    <motion.div
                                        key="empty"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        className="py-10 text-center text-slate-500 text-sm"
                                    >
                                        Wait for data...
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </motion.div>

                        <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.3 }}
                            className="glass-card rounded-3xl p-6"
                        >
                            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                <CloudRain className="w-5 h-5 text-sky-400" />
                                Live Factors (Mock)
                            </h3>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-white/5 rounded-xl p-3 flex flex-col items-center justify-center border border-white/5">
                                    <Sun className="w-6 h-6 text-amber-400 mb-2" />
                                    <span className="text-xl font-bold text-slate-200">24°C</span>
                                    <span className="text-xs text-slate-400">Temp</span>
                                </div>
                                <div className="bg-white/5 rounded-xl p-3 flex flex-col items-center justify-center border border-white/5">
                                    <Droplets className="w-6 h-6 text-blue-400 mb-2" />
                                    <span className="text-xl font-bold text-slate-200">65%</span>
                                    <span className="text-xs text-slate-400">Humidity</span>
                                </div>
                                <div className="bg-white/5 rounded-xl p-3 flex flex-col items-center justify-center border border-white/5">
                                    <Wind className="w-6 h-6 text-slate-400 mb-2" />
                                    <span className="text-xl font-bold text-slate-200">12km/h</span>
                                    <span className="text-xs text-slate-400">Wind</span>
                                </div>
                                <div className="bg-white/5 rounded-xl p-3 flex flex-col items-center justify-center border border-white/5">
                                    <Activity className="w-6 h-6 text-purple-400 mb-2" />
                                    <span className="text-xl font-bold text-slate-200">Normal</span>
                                    <span className="text-xs text-slate-400">Grid Status</span>
                                </div>
                            </div>
                        </motion.div>

                        {/* Load Breakdown */}
                        {breakdownData.length > 0 && (
                            <motion.div
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.4 }}
                                className="glass-card rounded-3xl p-6"
                            >
                                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                    <Clock className="w-5 h-5 text-amber-400" />
                                    Day Part Analysis
                                </h3>
                                <div className="h-40 w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={breakdownData} layout="vertical" margin={{ left: 0 }}>
                                            <XAxis type="number" hide />
                                            <YAxis dataKey="name" type="category" width={100} tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                                                cursor={{ fill: 'transparent' }}
                                            />
                                            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                                                {breakdownData.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </motion.div>
                        )}


                        <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.5 }}
                            className="glass-card rounded-3xl p-6"
                        >
                            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                <Terminal className="w-5 h-5 text-slate-400" />
                                System Logs
                            </h3>
                            <div className="h-[200px] overflow-y-auto pr-2 custom-scrollbar font-mono text-xs space-y-2">
                                {logs.map((log, i) => (
                                    <div key={i} className="flex gap-2 text-slate-400 border-b border-slate-800/50 pb-1 last:border-0">
                                        <span className="text-slate-600 shrink-0">[{format(log.timestamp, 'HH:mm:ss')}]</span>
                                        <span className={clsx(
                                            log.type === 'error' ? 'text-red-400' :
                                                log.type === 'warning' ? 'text-amber-400' :
                                                    log.type === 'success' ? 'text-emerald-400' : 'text-slate-300'
                                        )}>
                                            {log.message}
                                        </span>
                                    </div>
                                ))}
                                {logs.length === 0 && <span className="text-slate-600 italic">No logs yet...</span>}
                            </div>
                        </motion.div>
                        <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.3 }}
                            className="glass-card rounded-3xl p-6"
                        >
                            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                <Calendar className="w-5 h-5 text-cyan-400" />
                                System Config
                            </h3>
                            <div className="space-y-3 text-sm">
                                <div className="flex justify-between py-2 border-b border-slate-700/30">
                                    <span className="text-slate-400">Model Architecture</span>
                                    <span className="text-slate-200 font-medium">Transformer (TSFM)</span>
                                </div>
                                <div className="flex justify-between py-2 border-b border-slate-700/30">
                                    <span className="text-slate-400">Input Corpus</span>
                                    <span className="text-slate-200 font-medium">168 Hours</span>
                                </div>
                                <div className="flex justify-between py-2 border-b border-slate-700/30">
                                    <span className="text-slate-400">Inference Mode</span>
                                    <span className="text-slate-200 font-medium overflow-hidden text-ellipsis whitespace-nowrap max-w-[120px]" title="Autoregressive (Recursive)">
                                        Autoregressive
                                    </span>
                                </div>
                                <div className="flex justify-between py-2 border-b border-slate-700/30">
                                    <span className="text-slate-400">Device</span>
                                    <span className="text-indigo-400 font-bold tracking-wider">CUDA / GPU</span>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                </div>
            </main>
        </div>
    )
}

// Sub-components

function StatusIndicator({ label, active, loading }: { label: string; active: boolean; loading?: boolean }) {
    return (
        <div className="flex items-center gap-2">
            <div className="relative flex h-2.5 w-2.5">
                {loading ? (
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-slate-400 opacity-75"></span>
                ) : active ? (
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                ) : (
                    <span className="absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                )}
                <span className={clsx(
                    "relative inline-flex rounded-full h-2.5 w-2.5",
                    loading ? "bg-slate-500" : active ? "bg-emerald-500" : "bg-red-500"
                )}></span>
            </div>
            <span className={clsx("text-xs font-semibold tracking-wide", active ? "text-slate-300" : "text-slate-500")}>
                {label}
            </span>
        </div>
    )
}

function StatRow({ label, value, color }: { label: string; value: string; color: string }) {
    return (
        <div className="flex justify-between items-center p-3 rounded-xl bg-white/5 border border-white/5">
            <span className="text-slate-400 text-sm">{label}</span>
            <span className={clsx("font-bold text-lg", color)}>{value}</span>
        </div>
    )
}

function CustomTooltip({ active, payload }: any) {
    if (active && payload && payload.length) {
        return (
            <div className="glass-card p-3 rounded-xl border border-indigo-500/20 shadow-xl">
                <p className="text-slate-400 text-xs mb-1">{payload[0].payload.fullDate}</p>
                <p className="text-indigo-400 font-bold text-lg">
                    {payload[0].value.toLocaleString()} <span className="text-xs text-indigo-300/70 font-normal">MW</span>
                </p>
            </div>
        );
    }
    return null;
}

export default App
