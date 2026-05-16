import { useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';

/** Reads a number from search params, returns defaultVal if missing/invalid */
export function getNumParam(params: URLSearchParams, key: string, defaultVal: number): number {
  const v = params.get(key);
  if (v === null) return defaultVal;
  const n = parseFloat(v);
  return isFinite(n) ? n : defaultVal;
}

/** Reads a string from search params */
export function getStrParam(params: URLSearchParams, key: string, defaultVal: string): string {
  return params.get(key) ?? defaultVal;
}

/** Reads a boolean from search params (stored as '1'/'0') */
export function getBoolParam(params: URLSearchParams, key: string, defaultVal: boolean): boolean {
  const v = params.get(key);
  if (v === null) return defaultVal;
  return v === '1';
}

/** Hook that provides a function to copy the current URL (with new params) to clipboard */
export function useShareUrl() {
  const [, setSearchParams] = useSearchParams();

  const shareUrl = useCallback((params: Record<string, string | number | boolean>) => {
    const sp = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (typeof v === 'boolean') sp.set(k, v ? '1' : '0');
      else if (typeof v === 'number') sp.set(k, String(Math.round(v * 1e6) / 1e6));
      else sp.set(k, String(v));
    }
    setSearchParams(sp, { replace: true });
    const url = window.location.href;
    void navigator.clipboard.writeText(url).catch(() => {});
    return url;
  }, [setSearchParams]);

  return { shareUrl };
}
