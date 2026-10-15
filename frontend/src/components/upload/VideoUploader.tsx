import { useRef, useState, type DragEvent, type ChangeEvent } from 'react';
import { UploadCloud, X, Film, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

const ACCEPTED_TYPES = ['video/mp4', 'video/quicktime', 'video/webm'];
const ACCEPTED_EXTENSIONS = '.mp4, .mov, .webm';

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

interface VideoUploaderProps {
  file: File | null;
  onFileSelect: (file: File) => void;
  onRemove: () => void;
  maxUploadMB?: number;
  disabled?: boolean;
}

export function VideoUploader({ file, onFileSelect, onRemove, maxUploadMB = 500, disabled = false }: VideoUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function validate(f: File): string | null {
    if (!ACCEPTED_TYPES.includes(f.type) && !/\.(mp4|mov|webm)$/i.test(f.name)) {
      return `Unsupported format "${f.type || f.name}". Please upload MP4, MOV, or WebM.`;
    }
    if (f.size === 0) return 'The selected video is empty.';
    if (f.size > maxUploadMB * 1024 * 1024) {
      return `File is too large (${formatBytes(f.size)}). Maximum allowed size is ${maxUploadMB} MB.`;
    }
    return null;
  }

  function handleFile(f: File) {
    if (disabled) return;
    const err = validate(f);
    if (err) {
      setError(err);
      return;
    }
    setError(null);
    onFileSelect(f);
  }

  function onDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(true);
  }

  function onDragLeave(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFile(dropped);
  }

  function onInputChange(e: ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0];
    if (selected) handleFile(selected);
    // Reset so selecting the same file again still fires onChange
    e.target.value = '';
  }

  if (file) {
    return (
      <div className="rounded-xl border border-[#1e2d45] bg-[#131d35] p-5 flex items-start gap-4">
        <div className="h-10 w-10 rounded-lg bg-primary/20 flex items-center justify-center shrink-0">
          <Film className="h-5 w-5 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-slate-100 truncate">{file.name}</p>
          <p className="text-xs text-slate-400 mt-0.5">{formatBytes(file.size)}</p>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => {
            setError(null);
            onRemove();
          }}
          aria-label="Remove video"
          disabled={disabled}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        aria-label="Upload video — click or drag and drop"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => !disabled && e.key === 'Enter' && inputRef.current?.click()}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={cn(
          'cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-colors',
          dragging
            ? 'border-primary bg-primary/5'
            : 'border-[#1e2d45] hover:border-primary/50 hover:bg-primary/5',
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS}
          className="hidden"
          onChange={onInputChange}
          disabled={disabled}
        />
        <UploadCloud className="mx-auto h-10 w-10 text-slate-500 mb-3" />
        <p className="text-sm font-medium text-slate-200">
          Drag &amp; drop a video here, or{' '}
          <span className="text-primary underline">click to browse</span>
        </p>
        <p className="mt-1 text-xs text-slate-500">
          Supported formats: MP4, MOV, WebM — max {maxUploadMB} MB
        </p>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
