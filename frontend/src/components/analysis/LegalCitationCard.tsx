import { useState } from 'react';
import { ExternalLink, Copy, ChevronDown, ChevronUp, Check } from 'lucide-react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { DemoBadge } from '@/components/common/DemoBadge';
import type { TrafficLaw } from '@/types/law';

interface LegalCitationCardProps {
  law: TrafficLaw;
}

export function LegalCitationCard({ law }: LegalCitationCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const relevancePct = law.relevanceScore !== undefined ? Math.round(law.relevanceScore * 100) : null;

  function relevanceColor(score: number): string {
    if (score >= 85) return 'bg-green-500';
    if (score >= 65) return 'bg-amber-400';
    return 'bg-slate-500';
  }

  async function copyCitation() {
    const citation = `${law.title}, ${law.section} (${law.jurisdiction})`;
    await navigator.clipboard.writeText(citation).catch(() => undefined);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-start gap-2">
          <div className="flex-1 min-w-0">
            <p className="text-xs font-mono text-slate-500 mb-0.5">{law.section}</p>
            <h3 className="font-semibold text-slate-100 text-sm leading-snug">{law.title}</h3>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Badge variant="secondary" className="text-xs">
              {law.jurisdiction}
            </Badge>
            <DemoBadge />
          </div>
        </div>

        {relevancePct !== null && (
          <div className="mt-2">
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-slate-400">Relevance</span>
              <span className="text-slate-200">{relevancePct}%</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-[#0d1525] overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${relevanceColor(relevancePct)}`}
                style={{ width: `${relevancePct}%` }}
              />
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-3">
        {/* Excerpt */}
        <blockquote className="border-l-2 border-primary pl-3 italic text-sm text-slate-300">
          "{law.excerpt}"
        </blockquote>

        {/* Why it applies */}
        {law.whyApplies && (
          <div className="rounded-lg bg-primary/10 border border-primary/20 px-4 py-3">
            <p className="text-xs font-semibold text-primary uppercase tracking-wider mb-1">
              Why it applies
            </p>
            <p className="text-sm text-slate-300">{law.whyApplies}</p>
          </div>
        )}

        {/* Full text toggle */}
        {law.fullText && (
          <div>
            <Button
              variant="ghost"
              size="sm"
              className="text-xs gap-1 px-0 h-auto"
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded ? (
                <>
                  <ChevronUp className="h-3 w-3" /> Hide full text
                </>
              ) : (
                <>
                  <ChevronDown className="h-3 w-3" /> Show full text
                </>
              )}
            </Button>
            {expanded && (
              <pre className="mt-2 rounded-lg bg-[#0d1525] border border-[#1e2d45] p-4 text-xs text-slate-400 whitespace-pre-wrap font-mono leading-relaxed overflow-x-auto">
                {law.fullText}
              </pre>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2 pt-1">
          <Button
            variant="outline"
            size="sm"
            className="text-xs gap-1.5 h-8"
            onClick={copyCitation}
          >
            {copied ? (
              <>
                <Check className="h-3 w-3 text-green-400" /> Copied
              </>
            ) : (
              <>
                <Copy className="h-3 w-3" /> Copy citation
              </>
            )}
          </Button>
          <a
            href={law.sourceUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-auto inline-flex items-center gap-1.5 text-xs text-primary hover:underline"
          >
            Official source <ExternalLink className="h-3 w-3" />
          </a>
          {law.effectiveDate && (
            <span className="text-xs text-slate-500">Effective: {law.effectiveDate}</span>
          )}
        </div>

        {/* Topics */}
        {law.topics.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {law.topics.map((topic) => (
              <Badge key={topic} variant="outline" className="text-xs px-2 py-0">
                {topic}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
