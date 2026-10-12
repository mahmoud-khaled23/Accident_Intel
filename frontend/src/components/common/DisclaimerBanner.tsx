import { ShieldAlert } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { cn } from '@/lib/utils';

interface DisclaimerBannerProps {
  className?: string;
  message?: string;
}

const DEFAULT_MESSAGE =
  'This analysis is AI-assisted decision support only. It is not a final legal determination, official insurance decision, or substitute for professional legal advice. All findings should be independently verified by qualified professionals before any action is taken.';

export function DisclaimerBanner({ className, message = DEFAULT_MESSAGE }: DisclaimerBannerProps) {
  return (
    <Alert variant="warning" className={cn('', className)}>
      <ShieldAlert className="h-4 w-4" />
      <AlertDescription className="text-amber-300 font-medium">
        <span className="font-semibold">AI Analysis Disclaimer — </span>
        {message}
      </AlertDescription>
    </Alert>
  );
}
