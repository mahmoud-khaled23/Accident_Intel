import { useRef, useState, type DragEvent, type ChangeEvent } from 'react';
import { UploadCloud, X, ImageIcon, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

const ACCEPTED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
const ACCEPTED_EXTENSIONS = '.jpg,.jpeg,.png,.webp';
const MAX_BYTES = 20 * 1024 * 1024; // 20 MB
const MAX_FILES = 10;

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface ImageUploaderProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
}

export function ImageUploader({ files, onFilesChange }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);

  function validate(fileList: File[]): { valid: File[]; errs: string[] } {
    const valid: File[] = [];
    const errs: string[] = [];

    for (const f of fileList) {
      if (!ACCEPTED_TYPES.includes(f.type)) {
        errs.push(`"${f.name}" is not a supported format (JPG, PNG, WebP only).`);
        continue;
      }
      if (f.size > MAX_BYTES) {
        errs.push(`"${f.name}" exceeds 20 MB (${formatBytes(f.size)}).`);
        continue;
      }
      valid.push(f);
    }
    return { valid, errs };
  }

  function handleFiles(incoming: File[]) {
    const remaining = MAX_FILES - files.length;
    if (remaining <= 0) {
      setErrors([`Maximum ${MAX_FILES} images reached.`]);
      return;
    }
    const { valid, errs } = validate(incoming.slice(0, remaining));
    setErrors(errs);
    if (valid.length > 0) {
      onFilesChange([...files, ...valid]);
    }
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
    handleFiles(Array.from(e.dataTransfer.files));
  }

  function onInputChange(e: ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files ? Array.from(e.target.files) : [];
    if (selected.length > 0) handleFiles(selected);
    e.target.value = '';
  }

  function removeFile(index: number) {
    const next = [...files];
    next.splice(index, 1);
    onFilesChange(next);
  }

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload images — click or drag and drop"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
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
          multiple
          className="hidden"
          onChange={onInputChange}
        />
        <UploadCloud className="mx-auto h-10 w-10 text-slate-500 mb-3" />
        <p className="text-sm font-medium text-slate-200">
          Drag &amp; drop images here, or{' '}
          <span className="text-primary underline">click to browse</span>
        </p>
        <p className="mt-1 text-xs text-slate-500">
          JPG, JPEG, PNG, WebP — up to {MAX_FILES} images, max 20 MB each
        </p>
        {files.length > 0 && (
          <p className="mt-2 text-xs text-primary">
            {files.length} / {MAX_FILES} image{files.length !== 1 ? 's' : ''} selected
          </p>
        )}
      </div>

      {/* Error messages */}
      {errors.length > 0 && (
        <div className="space-y-1">
          {errors.map((err, i) => (
            <div
              key={i}
              className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-400"
            >
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <span>{err}</span>
            </div>
          ))}
        </div>
      )}

      {/* Thumbnails */}
      {files.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
          {files.map((file, idx) => (
            <div key={idx} className="relative group rounded-lg overflow-hidden border border-[#1e2d45] bg-[#0d1525]">
              <img
                src={URL.createObjectURL(file)}
                alt={`Preview ${idx + 1}`}
                className="w-full aspect-square object-cover"
              />
              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-1.5">
                <span className="text-xs text-slate-200 truncate flex-1">{file.name}</span>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="absolute top-1 right-1 h-6 w-6 bg-black/60 hover:bg-red-500/80 text-white opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={(e) => {
                  e.stopPropagation();
                  removeFile(idx);
                }}
                aria-label={`Remove image ${idx + 1}`}
              >
                <X className="h-3 w-3" />
              </Button>
              {/* Slot label */}
              <div className="absolute top-1 left-1">
                <span className="inline-flex items-center gap-0.5 rounded bg-black/60 px-1 py-0.5 text-[10px] text-slate-300">
                  <ImageIcon className="h-2.5 w-2.5" />
                  {idx + 1}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
