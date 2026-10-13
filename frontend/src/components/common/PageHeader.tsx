import React from 'react';
import { Link } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Breadcrumb {
  label: string;
  href?: string;
}

interface PageHeaderProps {
  title: string;
  description?: string;
  breadcrumbs?: Breadcrumb[];
  actions?: React.ReactNode;
  className?: string;
}

export function PageHeader({
  title,
  description,
  breadcrumbs,
  actions,
  className,
}: PageHeaderProps) {
  return (
    <div className={cn('mb-6', className)}>
      {breadcrumbs && breadcrumbs.length > 0 && (
        <nav className="mb-2 flex flex-wrap items-center gap-1 text-xs text-slate-500" aria-label="Breadcrumb">
          {breadcrumbs.map((crumb, idx) => (
            <React.Fragment key={idx}>
              {idx > 0 && <ChevronRight className="h-3 w-3 shrink-0" />}
              {crumb.href ? (
                <Link
                  to={crumb.href}
                  className="hover:text-slate-300 transition-colors"
                >
                  {crumb.label}
                </Link>
              ) : (
                <span className="text-slate-400 break-all">{crumb.label}</span>
              )}
            </React.Fragment>
          ))}
        </nav>
      )}
      <div className="flex flex-col sm:flex-row items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold text-slate-100 leading-tight break-words">{title}</h1>
          {description && (
            <p className="mt-1 text-sm text-slate-400 break-words">{description}</p>
          )}
        </div>
        {actions && (
          <div className="flex max-w-full flex-wrap items-center gap-2">{actions}</div>
        )}
      </div>
    </div>
  );
}
