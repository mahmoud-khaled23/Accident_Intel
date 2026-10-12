import { Link } from 'react-router-dom';
import { AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function NotFoundPage() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center text-center px-4">
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-[#1a2235] mb-6">
        <AlertTriangle className="h-8 w-8 text-warning" />
      </div>
      <h1 className="text-3xl font-semibold text-slate-100 mb-2">404 — Page Not Found</h1>
      <p className="text-slate-400 mb-8 max-w-md">
        The page you're looking for doesn't exist or has been moved.
      </p>
      <Button asChild>
        <Link to="/dashboard">Return to Dashboard</Link>
      </Button>
    </div>
  );
}
