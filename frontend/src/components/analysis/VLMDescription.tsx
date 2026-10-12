import { useState } from 'react';
import {
  Clock,
  Eye,
  HelpCircle,
  Car,
  TrafficCone,
  MapPin,
  Users,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Play,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { DemoBadge } from '@/components/common/DemoBadge';
import { LoadingState } from '@/components/common/LoadingState';
import type { VLMAnalysis } from '@/types/accident';

interface VLMDescriptionProps {
  description: VLMAnalysis | null;
  loading: boolean;
  onSeek?: (timestamp: number) => void;
}

function formatTs(s: number): string {
  return `${Math.floor(s / 60).toString().padStart(2, '0')}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
}

interface ExpandableProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

function ExpandableSection({ title, icon, children, defaultOpen = false }: ExpandableProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Card>
      <button
        className="w-full flex items-center gap-3 p-5 text-left"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="text-primary">{icon}</span>
        <span className="font-semibold text-slate-100 flex-1">{title}</span>
        {open ? (
          <ChevronUp className="h-4 w-4 text-slate-400" />
        ) : (
          <ChevronDown className="h-4 w-4 text-slate-400" />
        )}
      </button>
      {open && <CardContent className="pt-0">{children}</CardContent>}
    </Card>
  );
}

export function VLMDescription({ description, loading, onSeek }: VLMDescriptionProps) {
  if (loading) {
    return (
      <div className="py-16">
        <LoadingState message="Running Vision-Language Model analysis…" />
      </div>
    );
  }
  if (!description) return null;

  const facts = description.timeline.filter((e) => e.type === 'fact');
  const interpretations = description.timeline.filter((e) => e.type === 'interpretation');

  return (
    <div className="space-y-4">
      {/* Summary */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <CardTitle>Accident Summary</CardTitle>
            <DemoBadge className="ml-auto" />
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-slate-300 leading-relaxed">{description.summary}</p>
        </CardContent>
      </Card>

      {/* Event Timeline */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-primary" /> Chronological Event Timeline
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ol className="space-y-4">
            {description.timeline.map((event, i) => (
              <li key={i} className="flex gap-3">
                <div className="flex flex-col items-center">
                  <div
                    className={`h-3 w-3 rounded-full mt-1 shrink-0 ${
                      event.type === 'fact' ? 'bg-primary' : 'bg-amber-400'
                    }`}
                  />
                  {i < description.timeline.length - 1 && (
                    <div className="flex-1 w-px bg-[#1e2d45] mt-1" />
                  )}
                </div>
                <div className="pb-4 flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    {event.timestamp !== undefined && (
                      <button
                        className="flex items-center gap-1 text-xs font-mono text-primary hover:underline"
                        onClick={() => onSeek?.(event.timestamp!)}
                        aria-label={`Seek to ${formatTs(event.timestamp)}`}
                      >
                        <Play className="h-2.5 w-2.5" />
                        {formatTs(event.timestamp)}
                      </button>
                    )}
                    <Badge
                      variant={event.type === 'fact' ? 'default' : 'warning'}
                      className="text-xs"
                    >
                      {event.type === 'fact' ? (
                        <><Eye className="h-2.5 w-2.5" /> Observed</>
                      ) : (
                        <><HelpCircle className="h-2.5 w-2.5" /> Interpreted</>
                      )}
                    </Badge>
                  </div>
                  <p
                    className={`text-sm leading-relaxed ${
                      event.type === 'fact'
                        ? 'border-l-2 border-primary pl-3 text-slate-200'
                        : 'border-l-2 border-amber-400/60 pl-3 text-slate-300'
                    }`}
                  >
                    {event.content}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </CardContent>
      </Card>

      {/* Facts vs Interpretations */}
      <div className="grid md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <Eye className="h-4 w-4 text-primary" /> Observed Facts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {facts.map((ev, i) => (
                <li key={i} className="text-sm text-slate-300 border-l-2 border-primary pl-3">
                  {ev.timestamp !== undefined && (
                    <span className="text-xs font-mono text-primary mr-2">
                      {formatTs(ev.timestamp)}
                    </span>
                  )}
                  {ev.content}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <HelpCircle className="h-4 w-4 text-amber-400" /> Uncertain Interpretations
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {interpretations.map((ev, i) => (
                <li key={i} className="text-sm text-slate-300 border-l-2 border-amber-400/60 pl-3">
                  {ev.timestamp !== undefined && (
                    <span className="text-xs font-mono text-amber-400 mr-2">
                      {formatTs(ev.timestamp)}
                    </span>
                  )}
                  {ev.content}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>

      {/* Expandable sections */}
      <ExpandableSection
        title="Vehicle Actions"
        icon={<Car className="h-4 w-4" />}
        defaultOpen
      >
        <div className="space-y-3">
          {Object.entries(description.vehicleActions).map(([id, text]) => (
            <div key={id} className="rounded-lg border border-[#1e2d45] bg-[#0d1525] p-4">
              <p className="text-xs font-semibold text-primary uppercase tracking-wider mb-1">
                {id.toUpperCase()}
              </p>
              <p className="text-sm text-slate-300">{text}</p>
            </div>
          ))}
        </div>
      </ExpandableSection>

      <ExpandableSection title="Collision Description" icon={<AlertTriangle className="h-4 w-4" />}>
        <p className="text-sm text-slate-300 leading-relaxed">{description.collisionDescription}</p>
      </ExpandableSection>

      <ExpandableSection title="Traffic Signals &amp; Road Context" icon={<TrafficCone className="h-4 w-4" />}>
        <div className="space-y-3">
          <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Traffic Signals
            </p>
            <p className="text-sm text-slate-300">{description.trafficSignals}</p>
          </div>
          <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Road &amp; Intersection Context
            </p>
            <p className="text-sm text-slate-300">{description.roadContext}</p>
          </div>
        </div>
      </ExpandableSection>

      {description.pedestrians && (
        <ExpandableSection title="Pedestrian Observations" icon={<Users className="h-4 w-4" />}>
          <p className="text-sm text-slate-300">{description.pedestrians}</p>
        </ExpandableSection>
      )}

      <ExpandableSection title="Uncertainties &amp; Limitations" icon={<MapPin className="h-4 w-4" />}>
        <ul className="space-y-2">
          {description.uncertainties.map((u, i) => (
            <li key={i} className="flex gap-2 text-sm text-amber-300">
              <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-amber-400" />
              {u}
            </li>
          ))}
        </ul>
      </ExpandableSection>
    </div>
  );
}
