import { createBrowserRouter, Navigate } from 'react-router-dom';
import { AppShell } from '@/components/layout/AppShell';
import { DashboardPage } from '@/pages/DashboardPage';
import { NewAnalysisPage } from '@/pages/NewAnalysisPage';
import { AnalysisDetailsPage } from '@/pages/AnalysisDetailsPage';
import { AnalysisHistoryPage } from '@/pages/AnalysisHistoryPage';
import { DamageEstimatorPage } from '@/pages/DamageEstimatorPage';
import { TrafficLawsPage } from '@/pages/TrafficLawsPage';
import { SettingsPage } from '@/pages/SettingsPage';
import { NotFoundPage } from '@/pages/NotFoundPage';

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
        element: <DashboardPage />,
      },
      {
        path: 'analysis/new',
        element: <NewAnalysisPage />,
      },
      {
        path: 'analysis/:id',
        element: <AnalysisDetailsPage />,
      },
      {
        path: 'history',
        element: <AnalysisHistoryPage />,
      },
      {
        path: 'damage',
        element: <DamageEstimatorPage />,
      },
      {
        path: 'laws',
        element: <TrafficLawsPage />,
      },
      {
        path: 'settings',
        element: <SettingsPage />,
      },
      {
        path: '*',
        element: <NotFoundPage />,
      },
    ],
  },
]);
