import { AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ErrorStateProps {
  message?: string;
  onRetry?: () => void;
  className?: string;
}

export function ErrorState({
  message = 'Something went wrong.',
  onRetry,
  className,
}: ErrorStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center py-16 px-6 text-center',
        className,
      )}
      role="alert"
    >
      <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-red-500/10 border border-red-500/30">
        <AlertCircle className="h-7 w-7 text-red-400" />
      </div>
      <h3 className="text-base font-semibold text-slate-200">An error occurred</h3>
      <p className="mt-1 max-w-sm text-sm text-slate-400">{message}</p>
      {onRetry && (
        <Button variant="outline" size="sm" className="mt-4" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  );
}
