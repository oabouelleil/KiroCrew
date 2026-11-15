import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react'

/**
 * A usage-limit card the chat can resume offers "Switch to <harness> and continue"
 * for each other harness that can start, never the one that hit the limit, and only
 * with the Agent Harness preview on. A press asks the gateway to switch and continue;
 * a refusal or a continue that could not start is shown on the card.
 */

const client = vi.hoisted(() => ({
  acpBackends: vi.fn(),
  chatSlotSelectionCapabilities: vi.fn(),
  limitSwitch: vi.fn(),
  armLimitContinue: vi.fn(),
  cancelLimitContinue: vi.fn(),
}))
vi.mock('../api/client', () => ({ api: client }))

import { ErrorCard } from '../pages/chat/ErrorCard'
import { chatEndsOnRow } from '../pages/chat/LimitSwitchRow'
import { sseSlots } from '../store/dashboardSlice'
import type { ChatSlot } from '../types'
import { PREVIEW_SESSION_HARNESS } from '../utils/previewFlags'
import { createTestStore, renderWithProviders } from './helpers'

const AHEAD = '2096-10-02T09:46:40.000Z'
const PROSE = "You've hit your session limit \u00b7 resets soon."

function row(id: string, extra: Record<string, unknown> = {}) {
  return {
    id, policy_id: id, selectable: true, independent_setup: true, installed: 'installed',
    missing_components: [], install_command: '', restart_required: false, ...extra,
  }
}

function card(meta: Record<string, unknown> = { kind: 'usage_limit', reset_at: AHEAD }, switchSlot: string | undefined = 's1') {
  const store = createTestStore()
  store.dispatch(sseSlots([{ key: 's1', messages: 1, running: false } as ChatSlot]))
  return renderWithProviders(
    <ErrorCard content={PROSE} meta={meta} onContinue={() => undefined} limitContinueSlot="s1" limitSwitchSlot={switchSlot} />,
    { store },
  )
}

beforeEach(() => {
  localStorage.setItem(PREVIEW_SESSION_HARNESS, '1')
  client.acpBackends.mockResolvedValue({
    backends: [row('kiro'), row('claude'), row('codex'), row('pi', { installed: 'missing' }), row('goose', { restart_required: true })],
  })
  client.chatSlotSelectionCapabilities.mockResolvedValue({ known: true, backend: 'claude' })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  localStorage.clear()
})

const switchButtons = () => screen.queryAllByTestId(/^error-card-limit-switch-(?!failed)/)

describe('ErrorCard — switch harness on a usage limit', () => {
  it.each([
    ['with a reset time ahead', { kind: 'usage_limit', reset_at: AHEAD }],
    ['with no reset time', { kind: 'usage_limit' }],
  ])('offers every other harness that can start, %s', async (_label, meta) => {
    card(meta)
    await waitFor(() => expect(switchButtons()).toHaveLength(2))
    expect(screen.getByTestId('error-card-limit-switch-kiro')).toHaveTextContent('Switch to Kiro')
    expect(screen.getByTestId('error-card-limit-switch-codex')).toHaveTextContent(/Switch to .*Codex.* and continue/)
    // Never the harness that hit the limit, nor one that cannot start.
    expect(screen.queryByTestId('error-card-limit-switch-claude')).toBeNull()
    expect(screen.queryByTestId('error-card-limit-switch-pi')).toBeNull()
    expect(screen.queryByTestId('error-card-limit-switch-goose')).toBeNull()
    expect(screen.getByTestId('error-card-continue')).toBeVisible()
  })

  it('switches and continues on a press', async () => {
    client.limitSwitch.mockResolvedValue({ ok: true, backend: 'codex', continued: 'resumed' })
    card()
    fireEvent.click(await screen.findByTestId('error-card-limit-switch-codex'))
    await waitFor(() => expect(client.limitSwitch).toHaveBeenCalledWith('s1', 'codex'))
    expect(client.limitSwitch).toHaveBeenCalledTimes(1)
  })

  it('shows the refusal', async () => {
    const { ApiError } = await import('../api/apiError')
    client.limitSwitch.mockRejectedValue(
      new ApiError(409, 'a turn is in flight', JSON.stringify({ error: 'a turn is in flight', code: 'turn_in_flight' })),
    )
    card()
    fireEvent.click(await screen.findByTestId('error-card-limit-switch-codex'))
    expect(await screen.findByTestId('error-card-limit-switch-failed')).toBeVisible()
  })

  it('shows a continue that could not start', async () => {
    client.limitSwitch.mockResolvedValue({
      ok: true, backend: 'codex', continued: '', warning: 'the harness switched, but the chat did not continue',
    })
    card()
    fireEvent.click(await screen.findByTestId('error-card-limit-switch-codex'))
    expect(await screen.findByTestId('error-card-limit-switch-failed')).toHaveTextContent('did not continue')
  })

  it('offers nothing with the preview off', async () => {
    localStorage.clear()
    card()
    await waitFor(() => expect(screen.getByTestId('error-card-continue')).toBeVisible())
    expect(switchButtons()).toHaveLength(0)
    expect(client.acpBackends).not.toHaveBeenCalled()
  })

  it('offers nothing on a card that is not the chat newest usage-limit row', async () => {
    const store = createTestStore()
    renderWithProviders(<ErrorCard content={PROSE} meta={{ kind: 'usage_limit' }} onContinue={() => undefined} />, { store })
    await waitFor(() => expect(screen.getByTestId('error-card-continue')).toBeVisible())
    expect(switchButtons()).toHaveLength(0)
  })

  it('offers nothing once the chat has moved past the row', async () => {
    card({ kind: 'usage_limit', reset_at: AHEAD }, undefined)
    await waitFor(() => expect(screen.getByTestId('error-card-continue')).toBeVisible())
    expect(switchButtons()).toHaveLength(0)
  })
})

describe('chatEndsOnRow', () => {
  const limit = { role: 'error', content: 'limit', cls: '' }
  it.each([
    ['nothing after it', [limit], true],
    ['an empty assistant row after it', [limit, { role: 'assistant', content: '', cls: '' }], true],
    ['a tool row after it', [limit, { role: 'tool', content: 'x', cls: '' }], true],
    ['a notice after it, such as a switch row', [limit, { role: 'assistant', content: 'Harness switched', cls: '' }], false],
    ['a message after it', [limit, { role: 'user', content: 'go on', cls: '' }], false],
  ])('%s', (_label, rows, expected) => {
    expect(chatEndsOnRow(rows, 0)).toBe(expected)
  })
})
