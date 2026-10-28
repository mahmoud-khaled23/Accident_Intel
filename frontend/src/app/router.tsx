import { createBrowserRouter, Navigate } from 'react-router-dom';
import { AppShell } from '@/components/layout/AppShell';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    children: [
      {
        index: true,
        element: <Navigate to="/dashboard" replace />,
      },
      {
        path: 'dashboard',
        lazy: async () => ({ Component: (await import('@/pages/DashboardPage')).DashboardPage }),
      },
      {
        path: 'analysis/new',
        lazy: async () => ({ Component: (await import('@/pages/NewAnalysisPage')).NewAnalysisPage }),
      },
      {
        path: 'analysis/:id',
        lazy: async () => ({ Component: (await import('@/pages/AnalysisDetailsPage')).AnalysisDetailsPage }),
      },
      {
        path: 'history',
        lazy: async () => ({ Component: (await import('@/pages/AnalysisHistoryPage')).AnalysisHistoryPage }),
      },
      {
        path: 'damage',
        lazy: async () => ({ Component: (await import('@/pages/DamageEstimatorPage')).DamageEstimatorPage }),
      },
      {
        path: 'laws',
        lazy: async () => ({ Component: (await import('@/pages/TrafficLawsPage')).TrafficLawsPage }),
      },
      {
        path: 'cost',
        lazy: async () => ({ Component: (await import('@/pages/CostAnalysisPage')).CostAnalysisPage }),
      },
      {
        path: 'settings',
        lazy: async () => ({ Component: (await import('@/pages/SettingsPage')).SettingsPage }),
      },
      {
        path: '*',
        lazy: async () => ({ Component: (await import('@/pages/NotFoundPage')).NotFoundPage }),
      },
    ],
  },
]);
