import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const badgeVariants = cva(
  'inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors',
  {
    variants: {
      variant: {
        default:
          'bg-primary/20 text-primary border border-primary/30',
        secondary:
          'bg-background-card text-slate-300 border border-border',
        success:
          'bg-green-500/15 text-green-400 border border-green-500/30',
        warning:
          'bg-amber-500/15 text-amber-400 border border-amber-500/30',
        danger:
          'bg-red-500/15 text-red-400 border border-red-500/30',
        destructive:
          'bg-red-500/15 text-red-400 border border-red-500/30',
        outline:
          'border border-border text-slate-300 bg-transparent',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
