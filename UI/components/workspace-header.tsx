"use client"

import { Button } from "@/components/ui/button"
import { Activity, LogOut, Sparkles, User } from "lucide-react"

interface WorkspaceHeaderProps {
  userName: string
  onLogout: () => void
}

export default function WorkspaceHeader({ userName, onLogout }: WorkspaceHeaderProps) {
  return (
    <header className="sticky top-0 z-50 border-b border-slate-200 bg-white/85 backdrop-blur-xl shadow-sm">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-blue-500 to-fuchsia-600 blur-md opacity-60" />
              <div className="relative flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-fuchsia-600 shadow-lg">
                <Activity className="h-6 w-6 text-white" />
              </div>
            </div>
            <div>
              <h2 className="bg-gradient-to-r from-slate-900 to-slate-600 bg-clip-text text-xl font-bold text-transparent">
                Disease Prediction AI
              </h2>
              <p className="text-xs font-medium text-slate-500">Unified Workspace</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 sm:flex">
              <Sparkles className="h-4 w-4 text-fuchsia-500" />
              <span className="text-sm font-medium text-slate-600">Upload, analyze, and predict in one place</span>
            </div>

            <div className="flex items-center gap-3 border-l border-slate-200 pl-3">
              <div className="hidden text-right sm:block">
                <p className="text-sm font-semibold text-slate-900">{userName || "Workspace User"}</p>
                <p className="text-xs text-slate-500">No login mode</p>
              </div>
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-fuchsia-600 shadow-md">
                <User className="h-5 w-5 text-white" />
              </div>
            </div>

            <Button
              onClick={onLogout}
              variant="outline"
              className="h-10 rounded-xl border-slate-200 px-4 shadow-sm transition-all hover:border-red-200 hover:bg-red-50 hover:text-red-600"
            >
              <LogOut className="mr-2 h-4 w-4" />
              Refresh
            </Button>
          </div>
        </div>
      </div>
    </header>
  )
}
