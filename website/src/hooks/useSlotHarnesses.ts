import { useQuery, type QueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { ACP_BACKEND_KIRO } from '../api/acpBackend'
import { PREVIEW_SESSION_HARNESS } from '../utils/previewFlags'
import { usePreviewFlag } from './usePreviewFlag'

/**
 * The harnesses a chat could move to, for the controls that switch it: whether the
 * Agent Harness feature preview is on, the harness the chat runs now and the ones
 * this machine can start.
 *
 * `current` is the slot's selection-capabilities `backend` (the harness its live
 * session runs, or a new one would start on) in the policy spelling, so the runtime
 * `""` reads as `kiro`; `undefined` until it is read. `startable` is the
 * `/api/acp-backends` probe narrowed to selectable, installed and needing no
 * restart. Nothing is fetched while the preview is off.
 */
export function useSlotHarnesses(slot: string) {
  const enabled = usePreviewFlag(PREVIEW_SESSION_HARNESS)
  const probeQ = useQuery({
    queryKey: ['acpBackends'],
    queryFn: () => api.acpBackends(),
    enabled,
    retry: false,
  })
  const capsQ = useQuery({
    queryKey: ['slot-selection-capabilities', slot],
    queryFn: () => api.chatSlotSelectionCapabilities(slot),
    enabled,
  })
  const runtime = capsQ.data?.backend
  const current = runtime === undefined ? undefined : runtime || ACP_BACKEND_KIRO
  const startable = (probeQ.data?.backends ?? []).filter(
    b => b.selectable && b.installed !== 'missing' && !b.restart_required,
  )
  return { enabled, current, startable }
}

/**
 * After a switch: show *backend* as the chat's harness at once, then re-read it. The
 * harness mark, the model list and the effort control all follow it.
 */
export async function showSlotHarness(queryClient: QueryClient, slot: string, backend: string) {
  const key = ['slot-selection-capabilities', slot]
  await queryClient.cancelQueries({ queryKey: key })
  queryClient.setQueryData<{ backend?: string }>(key, prev => (
    prev ? { ...prev, backend: backend === ACP_BACKEND_KIRO ? '' : backend } : prev
  ))
  void queryClient.invalidateQueries({ queryKey: key })
}
