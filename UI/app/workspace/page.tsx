"use client"

import WorkspaceDashboard from "@/components/workspace-dashboard"

export default function WorkspacePage() {
  const handleLogout = () => {
    if (typeof window !== "undefined") {
      window.location.reload()
    }
  }

  return <WorkspaceDashboard userName="Workspace" onLogout={handleLogout} />
}
