import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'
import SessionHarnessPicker from '../components/SessionHarnessPicker'
import { api } from '../api/client'
import { ApiError } from '../api/apiError'
import { i18nT } from '../i18n/t'
import { PREVIEW_SESSION_HARNESS } from '../utils/previewFlags'

/**
 * The model picker's harness row moves the chat onto another harness. It shows
 * only with the Agent Harness preview on, offers only harnesses that can start,
 * marks the one the chat runs, writes the pick, and is disabled while a reply runs.
 */

const SLOT = 'dashboard:1'

function row(id: string, extra: Record<string, unknown> = {}) {
  return {
    id, policy_id: id, selectable: true, independent_setup: true, installed: 'installed',
    missing_components: [], install_command: '', restart_required: false, ...extra,
  }
}

function mount(busy = false, runtime = 'claude') {
  vi.spyOn(api, 'acpBackends').mockResolvedValue({
    backends: [row('kiro'), row('claude'), row('codex'), row('pi', { installed: 'missing' }), row('goose', { restart_required: true })],
  } as never)
  const caps = vi.spyOn(api, 'chatSlotSelectionCapabilities').mockResolvedValue({ known: true, backend: runtime })
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={qc}><SessionHarnessPicker slot={SLOT} busy={busy} /></QueryClientProvider>)
  return { caps }
}

const select = () => screen.findByRole('combobox', { name: i18nT('components.sessionHarnessPicker.label') })

beforeEach(() => { localStorage.setItem(PREVIEW_SESSION_HARNESS, '1') })
afterEach(() => { vi.restoreAllMocks(); localStorage.clear() })

describe('SessionHarnessPicker', () => {
  it('offers the harnesses that can start and marks the current one', async () => {
    mount()
    const el = await select()
    await waitFor(() => expect((el as HTMLSelectElement).value).toBe('claude'))
    const values = Array.from((el as HTMLSelectElement).options).map(o => o.value)
    expect(values).toEqual(['kiro', 'claude', 'codex'])
  })

  it('reads Kiro CLI from its runtime spelling', async () => {
    mount(false, '')
    const el = await select()
    await waitFor(() => expect((el as HTMLSelectElement).value).toBe('kiro'))
  })

  it('writes the pick and re-reads the capabilities', async () => {
    const write = vi.spyOn(api, 'chatSlotBackend').mockResolvedValue({ ok: true, backend: 'codex' })
    const { caps } = mount()
    const el = await select()
    await waitFor(() => expect((el as HTMLSelectElement).value).toBe('claude'))
    const before = caps.mock.calls.length

    fireEvent.change(el, { target: { value: 'codex' } })

    await waitFor(() => expect(write).toHaveBeenCalledWith(SLOT, 'codex'))
    await waitFor(() => expect(caps.mock.calls.length).toBeGreaterThan(before))
  })

  it('shows a teardown warning the switch came back with', async () => {
    vi.spyOn(api, 'chatSlotBackend').mockResolvedValue({ ok: true, backend: 'codex', warning: 'old process may linger' })
    mount()
    const el = await select()
    await waitFor(() => expect((el as HTMLSelectElement).value).toBe('claude'))
    fireEvent.change(el, { target: { value: 'codex' } })
    expect(await screen.findByText('old process may linger')).toBeTruthy()
  })

  it('shows the refusal', async () => {
    vi.spyOn(api, 'chatSlotBackend').mockRejectedValue(
      new ApiError(400, "backend 'codex' is not installed", JSON.stringify({ code: 'backend_not_installed' })),
    )
    mount()
    const el = await select()
    await waitFor(() => expect((el as HTMLSelectElement).value).toBe('claude'))
    fireEvent.change(el, { target: { value: 'codex' } })
    expect(await screen.findByText(/not installed/)).toBeTruthy()
  })

  it('is disabled while a reply runs, and says why', async () => {
    mount(true)
    const el = await select()
    expect((el as HTMLSelectElement).disabled).toBe(true)
    expect(screen.getByText(i18nT('components.sessionHarnessPicker.busy'))).toBeTruthy()
  })

  it('renders nothing with the preview off', async () => {
    localStorage.clear()
    mount()
    expect(screen.queryByTestId('session-harness-picker')).toBeNull()
  })
})
