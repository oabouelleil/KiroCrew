import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ArrowLeftRight, Loader2 } from 'lucide-react'

import { api } from '../../api/client'
import { acpBackendName } from '../../api/acpBackend'
import ErrorNotice from '../../components/ErrorNotice'
import { showSlotHarness, useSlotHarnesses } from '../../hooks/useSlotHarnesses'
import { i18nT } from '../../i18n/t'
import { agentSwitchFailureMessage } from '../../utils/agentSwitchFeedback'
import type { ChatMessage } from '../../types'

const ACTION_BTN =
  'inline-flex items-center gap-2 text-[12px] leading-5 font-medium text-balance px-3 py-1 rounded-md border-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed transition-colors'

/**
 * Whether the chat still ends on the row at *index*: no user or assistant row with
 * content after it. The gateway's own test (`limit_continue.ends_on_usage_limit`),
 * so the card offers a switch only where the route would take it: a notice such as
 * a harness switch's boundary row ends the offer.
 */
export function chatEndsOnRow(messages: readonly ChatMessage[], index: number): boolean {
  for (let i = index + 1; i < messages.length; i++) {
    const row = messages[i]
    if ((row.role === 'user' || row.role === 'assistant') && row.content) return false
  }
  return true
}

/**
 * On a usage-limit card the chat can resume: one "Switch to <harness> and continue"
 * per other harness this machine can start (`POST /api/chat/slots/{slot}/limit-switch`).
 * The gateway moves the chat onto it and runs the same continue Resume runs. Shown
 * only while the Agent Harness feature preview is on, and never for the harness the
 * chat runs, which is the one that hit the limit.
 *
 * A refusal, or a switch whose continue could not start, is shown inline; a switch
 * that continued ends the card, since the chat ends on the switch's own row then.
 */
export function LimitSwitchRow({ slot, className = '' }: { slot: string; className?: string }) {
  const { enabled, current, startable } = useSlotHarnesses(slot)
  const queryClient = useQueryClient()
  const move = useMutation({
    mutationFn: (backend: string) => api.limitSwitch(slot, backend),
    onSuccess: (_result, backend) => showSlotHarness(queryClient, slot, backend),
  })
  if (!enabled || current === undefined) return null
  const targets = startable.filter(b => b.policy_id !== current)
  if (targets.length === 0) return null
  return (
    <div className={`flex flex-col gap-1 ${className}`} data-testid="error-card-limit-switch">
      <div className="flex flex-wrap items-center gap-2">
        {targets.map(b => (
          <button
            key={b.policy_id}
            type="button"
            onClick={() => move.mutate(b.policy_id)}
            disabled={move.isPending}
            className={`${ACTION_BTN} bg-transparent text-text ring-1 ring-inset ring-border hover:bg-bg-elevated`}
            data-testid={`error-card-limit-switch-${b.policy_id}`}
          >
            {move.isPending && move.variables === b.policy_id
              ? <Loader2 size={12} className="lucide-inline shrink-0 animate-spin" aria-hidden="true" />
              : <ArrowLeftRight size={12} className="lucide-inline shrink-0" aria-hidden="true" />}
            {i18nT('pages.chat.errorCard.limit_switch', { harness: acpBackendName(b) })}
          </button>
        ))}
      </div>
      <ErrorNotice
        variant="inline"
        message={move.isError ? agentSwitchFailureMessage(move.error) : (move.data?.warning ?? null)}
        noHandoffReason="This card exists because the chat's agent has no allowance left; an Ask-the-agent chat on the same harness would be refused the same way."
        testId="error-card-limit-switch-failed"
      />
    </div>
  )
}
