import { useId } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { acpBackendName } from '../api/acpBackend'
import { showSlotHarness, useSlotHarnesses } from '../hooks/useSlotHarnesses'
import { i18nT } from '../i18n/t'
import { agentSwitchFailureMessage } from '../utils/agentSwitchFeedback'
import ErrorNotice from './ErrorNotice'
import { NativeSelect } from './ui/native-select'

/**
 * The harness row in the composer's model picker: moves this chat onto another
 * agent harness (`POST /api/chat/slots/{slot}/backend`). Shown only while the
 * Agent Harness feature preview is on; the gateway refuses the write otherwise.
 *
 * The options are the harnesses this machine can start right now and the current
 * one is the harness the chat runs (`useSlotHarnesses`). A switch is refused while
 * a reply runs, so the control is disabled then and says why.
 */
export default function SessionHarnessPicker({ slot, busy }: { slot: string; busy: boolean }) {
  const { enabled, current, startable } = useSlotHarnesses(slot)
  const queryClient = useQueryClient()
  const hintId = useId()
  const write = useMutation({
    mutationFn: (backend: string) => api.chatSlotBackend(slot, backend),
    onSuccess: (_result, backend) => showSlotHarness(queryClient, slot, backend),
  })
  if (!enabled) return null
  // A chat on a harness that can no longer start still names it, so the control
  // never claims the chat runs on something else.
  const options = current && !startable.some(b => b.policy_id === current)
    ? [...startable, { id: current, policy_id: current }]
    : startable
  if (current === undefined || options.length < 2) return null
  const label = i18nT('components.sessionHarnessPicker.label')
  const value = write.isPending ? write.variables : current
  return (
    <div className="shrink-0 border-t border-border px-3 py-2.5" data-testid="session-harness-picker">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[12px] text-text">{label}</span>
        <NativeSelect
          aria-label={label}
          aria-describedby={hintId}
          value={value}
          disabled={busy}
          busy={write.isPending}
          aria-busy={write.isPending || undefined}
          onChange={e => {
            if (write.isPending || e.target.value === current) return
            write.mutate(e.target.value)
          }}
          wrapperStyle={{ minWidth: 0, flex: '0 1 60%' }}
        >
          {options.map(b => (
            <option key={b.policy_id} value={b.policy_id}>{acpBackendName(b)}</option>
          ))}
        </NativeSelect>
      </div>
      <p id={hintId} className="mt-1 text-[11px] leading-snug text-muted">
        {busy ? i18nT('components.sessionHarnessPicker.busy') : i18nT('components.sessionHarnessPicker.hint')}
      </p>
      {/* No hand-off: the picker sits over the composer, which may hold an unsent draft. */}
      <ErrorNotice
        className="mt-1.5"
        variant="inline"
        message={write.isError ? agentSwitchFailureMessage(write.error) : (write.data?.warning ?? null)}
      />
    </div>
  )
}
