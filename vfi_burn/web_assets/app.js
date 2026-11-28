/* vfi_burn Local Web UI — Vanilla JS Controller (Zero External Dependencies) */
(() => {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  const TOKEN = params.get("token") || sessionStorage.getItem("vfi_token") || "";
  if (TOKEN) {
    sessionStorage.setItem("vfi_token", TOKEN);
  }

  const state = {
    step: "target",
    overview: null,
    config: null,
    watchTimer: null,
    sse: null,
    burnRowStates: {},
    burnRunning: false,
    fileModal: {
      dir: "private",
      filter: "all",
      selected: null,
      onPick: null,
    },
    confirmModal: {
      expected: "",
      onResolve: null,
    },
  };

  const $ = (id) => document.getElementById(id);

  function esc(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  let toastTimer = null;
  function showToast(msg, kind = "") {
    const el = $("toast");
    el.textContent = msg;
    el.className = "toast show" + (kind ? " " + kind : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = "toast"; }, 4200);
  }

  async function api(method, path, body = null) {
    const opts = {
      method,
      headers: { "X-VFI-Token": TOKEN },
    };
    if (body !== null) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
    if (!res.ok || !data.ok) {
      throw new Error(data.error || `Request failed (${res.status})`);
    }
    return data.data;
  }

  // ------------------------------------------------------------------ Rendering Helpers
  function renderStepHelp(stepHelp) {
    if (!stepHelp) return;
    for (const [key, info] of Object.entries(stepHelp)) {
      const box = $(`help-${key}`);
      if (!box) continue;
      box.innerHTML = `
        <div><div class="help-label">What this step is for</div><div class="help-body">${esc(info.purpose)}</div></div>
        <div><div class="help-label">When to use it</div><div class="help-body">${esc(info.when)}</div></div>
        <div><div class="help-label">Bench tip</div><div class="help-body">${esc(info.tip)}</div></div>
      `;
    }
  }

  function renderBeforeAfterGrid(containerId, s) {
    const el = $(containerId);
    if (!el) return;
    if (!s) {
      el.innerHTML = `<div class="ba-card"><div class="ba-label">Before / After</div><div class="ba-value">--</div><div class="ba-sub">Load a card or plan to inspect expected hardware capabilities.</div></div>`;
      return;
    }
    const fmt = (n) => (n === null || n === undefined ? "?" : Number(n).toLocaleString());
    const ccChanged = s.cuda_cores_before !== s.cuda_cores_after;
    const fbpChanged = s.fbp_before !== s.fbp_after;
    const pcieChanged = s.pcie_before !== s.pcie_after;
    const changedFeats = (s.features || []).filter((f) => f.changed);
    const featSummary = changedFeats.length
      ? changedFeats.map((f) => `${f.name}: ${f.after}`).join(" · ")
      : `${s.pci_id_after} (${s.sku_after})`;
    const featSub = (s.features || [])
      .map((f) => `${f.name}: ${f.changed ? `${f.before} → ${f.after}` : f.after}`)
      .slice(0, 3)
      .join(" | ");

    el.innerHTML = `
      <div class="ba-card">
        <div class="ba-label">CUDA Cores / SMs</div>
        <div class="ba-value ${ccChanged ? "changed" : ""}">${fmt(s.cuda_cores_before)} → ${fmt(s.cuda_cores_after)}</div>
        <div class="ba-sub">${s.sm_before ?? "?"} → ${s.sm_after ?? "?"} SMs (healthy max ${s.sm_max_healthy ?? "?"} SMs / ${fmt(s.cuda_cores_max_healthy)} CUDA)</div>
      </div>
      <div class="ba-card">
        <div class="ba-label">Est. VRAM</div>
        <div class="ba-value ${fbpChanged || s.targeting_a100_vbios ? "changed" : ""}">${esc(s.vram_label)}</div>
        <div class="ba-sub">1 GB/FBP on CMP VBIOS · ${s.a100_gb_per_fbp} GB/FBP under A100 VBIOS</div>
      </div>
      <div class="ba-card">
        <div class="ba-label">Memory Bus Width</div>
        <div class="ba-value ${fbpChanged ? "changed" : ""}">${s.bus_width_before ?? "?"}-bit → ${s.bus_width_after ?? "?"}-bit</div>
        <div class="ba-sub">${s.fbp_before ?? "?"} → ${s.fbp_after ?? "?"}/12 FBPs (healthy max ${s.fbp_max_healthy ?? "?"})</div>
      </div>
      <div class="ba-card">
        <div class="ba-label">PCIe Link Version</div>
        <div class="ba-value ${pcieChanged ? "changed" : ""}">${esc(s.pcie_before)} → ${esc(s.pcie_after)}</div>
        <div class="ba-sub">Fused link speed &amp; boot caps</div>
      </div>
      <div class="ba-card">
        <div class="ba-label">Features &amp; Identity</div>
        <div class="ba-value ${changedFeats.length ? "changed" : ""}" style="font-size:13px;">${esc(featSummary)}</div>
        <div class="ba-sub">${esc(featSub)}</div>
      </div>
    `;
  }

  function renderOverview(ov) {
    state.overview = ov;
    renderStepHelp(ov.step_help);

    // Top chips
    $("chip-target").innerHTML = `Target: <strong class="mono">${esc(ov.target_label)}</strong>`;
    $("top-journal-meta").textContent = `Journal: ${ov.journal_path}`;

    const badge = ov.live_status?.badge || (ov.live_error ? "ERROR" : "NO TARGET");
    const hChip = $("chip-health");
    hChip.className = "chip " + (
      badge === "GOLDILOCKS" || badge === "ARMED" ? "ok" :
      badge === "POISON" || badge === "ERROR" ? "danger" :
      badge === "COLD" ? "info" : "warn"
    );
    hChip.innerHTML = `Card: <strong class="mono">${esc(badge)}</strong>`;

    const pChip = $("chip-plan");
    if (ov.plan) {
      pChip.className = "chip " + (ov.sku_mismatch ? "danger" : "info");
      pChip.innerHTML = `Plan: <strong class="mono">${esc(ov.plan.path)}</strong> (${ov.plan.record_count} rec${ov.sku_mismatch ? " · SKU MISMATCH!" : ""})`;
    } else if (ov.plan_error) {
      pChip.className = "chip danger";
      pChip.innerHTML = `Plan: <strong>INVALID</strong>`;
    } else {
      pChip.className = "chip";
      pChip.innerHTML = `Plan: <strong>none</strong>`;
    }

    const armBtn = $("btn-arm-toggle");
    if (ov.is_simulated) {
      armBtn.className = "btn btn-sm";
      armBtn.textContent = "Mode: Simulation";
    } else if (ov.execute_live) {
      armBtn.className = "btn btn-danger btn-sm";
      armBtn.textContent = "LIVE WRITES ARMED (Click to Disarm)";
    } else {
      armBtn.className = "btn btn-sm";
      armBtn.textContent = "Mode: Read-Only (Click to Arm)";
    }

    // Advisor bar
    $("advisor-headline").textContent = ov.headline;
    $("advisor-detail").textContent = ov.detail;
    $("btn-advisor-go").dataset.targetStep = ov.recommended_step;

    // Stepper dots
    document.querySelectorAll(".step-tab").forEach((btn) => {
      const s = btn.dataset.step;
      btn.dataset.state = (ov.step_states && ov.step_states[s]) || "pending";
    });

    // Inputs
    $("inp-journal-path").value = ov.journal_path || "";
    $("inp-receipt-path").value = ov.receipt_path || "";
    $("sel-confirm-mode").value = ov.confirm_mode || "group";

    // Driver blocker banner
    const db = ov.driver_blockers;
    const detachBtn = $("btn-detach-driver");
    const dbBanner = $("driver-blocker-banner");
    detachBtn.disabled = !(ov.bdf && !ov.is_simulated);
    if (db && db.needed) {
      dbBanner.style.display = "block";
      dbBanner.innerHTML = `<div class="alert warn">Live GPU <strong class="mono">${esc(db.bdf)}</strong> has driver blocker(s): bound=<code>${esc(db.bound_driver || "none")}</code>, loaded modules=<code>${esc((db.loaded_modules || []).join(", ") || "none")}</code>. Click <strong>Unload nvidia/nouveau &amp; Bind pci-stub</strong> before reading BAR0.</div>`;
    } else {
      dbBanner.style.display = "none";
    }

    // Overrides panel
    renderOverrides(ov.overrides);

    // If plan loaded and configurator not open, show plan's Before/After summary in Step 2
    if (!state.config && ov.plan) {
      renderBeforeAfterGrid("cfg-ba-grid", ov.plan.summary);
      $("cfg-card-subtitle").textContent = `Showing active loaded plan: ${ov.plan.path} (${ov.plan.pci_id} ${ov.plan.sku_name})`;
      refreshLoadedPlanReview();
    }
  }

  function setStep(step) {
    state.step = step;
    document.querySelectorAll(".step-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.step === step);
    });
    document.querySelectorAll(".step-view").forEach((view) => {
      view.classList.toggle("active", view.id === `view-${step}`);
    });
    if (step === "target") refreshGpus();
    if (step === "preflight") refreshPreflightStatus(false);
    if (step === "burn") refreshBurnPreview();
    if (step === "cold_verify") refreshColdPreflight();
    if (step === "tools") refreshJournal();
  }

  // ------------------------------------------------------------------ Step 1: Target & GPUs
  async function refreshGpus() {
    try {
      const gpus = await api("GET", "/api/gpus?probe=1");
      const tbody = $("gpu-table-body");
      if (!gpus.length) {
        tbody.innerHTML = `<tr><td colspan="8" style="color:var(--text-muted);">No PCI display/3D GPUs detected in sysfs. Use Browse Sim-PCIe/Fake Snapshot below for offline testing.</td></tr>`;
        return;
      }
      tbody.innerHTML = gpus.map((g) => {
        const smStr = g.cur_sm !== null ? `${g.cur_sm}/${g.max_sm} SMs (${(g.cur_cuda || 0).toLocaleString()} CUDA)` : "--";
        const memStr = g.cur_fbp !== null
          ? `${g.cur_fbp}/${g.max_fbp} FBPs · ${g.cur_bus_bits}b · ~${g.cur_vram_cmp_gb}GB${g.cur_vram_a100_gb ? `/${g.cur_vram_a100_gb}GB` : ""}`
          : "--";
        const stateTag = g.fuse_state || (g.probe_note ? g.probe_note : "--");
        const badgeCls = g.fuse_state === "GOLDILOCKS" ? "ok" : (g.fuse_state === "COLD" ? "info" : "");
        return `
          <tr>
            <td class="mono"><strong>${esc(g.bdf)}</strong></td>
            <td><span class="mono">${esc(g.pci_id)}</span> ${esc(g.name)}</td>
            <td class="mono">${esc(g.driver)}</td>
            <td class="mono">${esc(g.link_current)} (max ${esc(g.link_max)})</td>
            <td class="mono">${esc(smStr)}</td>
            <td class="mono">${esc(memStr)}</td>
            <td><span class="chip ${badgeCls}">${esc(stateTag)}</span></td>
            <td><button class="btn btn-sm ${g.selected ? "btn-primary" : ""}" data-select-bdf="${esc(g.bdf)}">${g.selected ? "Selected" : "Select"}</button></td>
          </tr>
        `;
      }).join("");
      tbody.querySelectorAll("[data-select-bdf]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          try {
            const ov = await api("POST", "/api/target/gpu", { bdf: btn.dataset.selectBdf });
            renderOverview(ov);
            refreshGpus();
            showToast(`Selected live GPU ${btn.dataset.selectBdf} (Read-Only mode).`, "ok");
          } catch (e) {
            showToast(e.message, "error");
          }
        });
      });
    } catch (e) {
      showToast(e.message, "error");
    }
  }

  // ------------------------------------------------------------------ Step 2: Configure & Die Map
  function renderConfigState(cfg) {
    state.config = cfg;
    if (!cfg || !cfg.loaded) {
      $("cfg-editor-wrap").style.display = "none";
      return;
    }
    $("cfg-editor-wrap").style.display = "flex";
    $("cfg-card-subtitle").textContent = `${cfg.pci_id} (${cfg.sku_name}) · Next free FF row: ${cfg.next_free_row} (${cfg.free_row_count} free slots)`;
    renderBeforeAfterGrid("cfg-ba-grid", cfg.summary);

    // Split Enable-All tiles
    $("tile-all-compute").classList.toggle("on", !!cfg.all_healthy_compute);
    $("chk-all-compute").textContent = cfg.all_healthy_compute ? "*" : "";
    $("tile-all-compute-sub").textContent = `${cfg.summary.sm_before} → ${cfg.summary.sm_after} SMs (healthy max ${cfg.summary.sm_max_healthy} SMs / ${(cfg.summary.cuda_cores_max_healthy || 0).toLocaleString()} CUDA)`;

    $("tile-all-memory").classList.toggle("on", !!cfg.all_healthy_memory);
    $("chk-all-memory").textContent = cfg.all_healthy_memory ? "*" : "";
    $("tile-all-memory-sub").textContent = `${cfg.summary.fbp_before} → ${cfg.summary.fbp_after}/12 FBPs (max ${cfg.summary.fbp_max_healthy}) · ${cfg.summary.vram_label}`;

    // GPC x TPC Grid
    const gpcGrid = $("gpc-grid");
    gpcGrid.innerHTML = cfg.gpcs.map((g) => {
      const gCls = g.defective ? "defective" : (
        g.target && !g.current ? "unlock" :
        !g.target && g.current ? "lock" :
        g.target ? "cur-on" : ""
      );
      const tpcsHtml = g.tpcs.map((t) => {
        const tCls = t.defective ? "defective" : (
          t.target && !t.current ? "unlock" :
          !t.target && t.current ? "lock" :
          t.target ? "cur-on" : ""
        );
        return `<button class="tpc-cell ${tCls}" ${t.defective ? "disabled" : ""} data-gpc="${g.index}" data-tpc="${t.index}" data-next="${!t.target}" title="GPC${g.index}/TPC${t.index} (2 SMs / 128 CUDA)">${t.defective ? "X" : `T${t.index}`}</button>`;
      }).join("");
      return `
        <div class="gpc-col">
          <button class="gpc-hdr ${gCls}" ${g.defective ? "disabled" : ""} data-toggle-gpc="${g.index}" data-next="${!g.target}" title="Toggle entire GPC${g.index}">GPC${g.index}</button>
          ${tpcsHtml}
        </div>
      `;
    }).join("");

    gpcGrid.querySelectorAll("[data-toggle-gpc]").forEach((btn) => {
      btn.addEventListener("click", () => patchConfig({
        action: "set_gpc", gpc: Number(btn.dataset.toggleGpc), enabled: btn.dataset.next === "true",
      }));
    });
    gpcGrid.querySelectorAll("[data-tpc]").forEach((btn) => {
      btn.addEventListener("click", () => patchConfig({
        action: "set_tpc", gpc: Number(btn.dataset.gpc), tpc: Number(btn.dataset.tpc), enabled: btn.dataset.next === "true",
      }));
    });

    // HBM / FBP Grid
    const a100Gb = cfg.summary.a100_gb_per_fbp || 4;
    $("hbm-map-sub").textContent = `Each FBP = +512-bit bus (+1 GB CMP VBIOS / +${a100Gb} GB A100 VBIOS).`;
    const hbmGrid = $("hbm-grid");
    hbmGrid.innerHTML = cfg.fbps.map((f) => {
      const fCls = f.defective ? "defective" : (
        f.target && !f.current ? "unlock" :
        !f.target && f.current ? "lock" :
        f.target ? "cur-on" : ""
      );
      const tag = f.defective ? "DEFECT" : (
        f.target && !f.current ? "UNLOCK" :
        !f.target && f.current ? "LOCK" :
        f.target ? "ON" : "OFF"
      );
      return `
        <button class="fbp-tile ${fCls}" ${f.defective ? "disabled" : ""} data-fbp="${f.index}" data-next="${!f.target}">
          <span class="mono"><strong>FBP${f.index}</strong> <span style="color:var(--text-muted);">S${f.stack}</span></span>
          <span class="mono" style="font-size:11px;">${tag}</span>
        </button>
      `;
    }).join("");
    hbmGrid.querySelectorAll("[data-fbp]").forEach((btn) => {
      btn.addEventListener("click", () => patchConfig({
        action: "set_fbp", fbp: Number(btn.dataset.fbp), enabled: btn.dataset.next === "true",
      }));
    });

    // PCIe Radio Stack
    const pcieOpts = [
      ["keep", `Keep current (${cfg.pcie_current})`],
      ["gen3", "PCIe Gen3 (2 FF rows: c2/b39, c2/b49)"],
      ["gen4", "PCIe Gen4 (4 FF rows: Gen3 + MAGIC_BITS_D)"],
      ["lock_gen12", "Lock to PCIe Gen1/Gen2"],
    ];
    $("pcie-radio-stack").innerHTML = pcieOpts.map(([val, lbl]) => `
      <div class="radio-row ${cfg.pcie_mode === val ? "selected" : ""}" data-pcie="${val}">
        <span class="mono">${cfg.pcie_mode === val ? "(*)" : "( )"}</span>
        <span>${esc(lbl)}</span>
      </div>
    `).join("");
    $("pcie-radio-stack").querySelectorAll("[data-pcie]").forEach((el) => {
      el.addEventListener("click", () => patchConfig({ action: "set_pcie_mode", mode: el.dataset.pcie }));
    });

    // Device ID Radio Stack
    const devOpts = [
      ["keep", `Keep current (${cfg.pci_id}) — no VBIOS change`],
      ["0x20f1", "A100 PCIe 40 GB (10de:20f1) — needs A100 VBIOS + 128 GB BAR1"],
      ["0x20b5", "A100 PCIe 80 GB (10de:20b5) — needs A100 VBIOS + 128 GB BAR1"],
      ["0x20c2", "Revert to CMP 170HX 8 GB (10de:20c2) — needs CMP 8 GB VBIOS"],
      ["0x2082", "Revert to CMP 170HX 10 GB (10de:2082) — needs CMP 10 GB VBIOS"],
    ];
    const curDev = cfg.a100_devid || "keep";
    $("devid-radio-stack").innerHTML = devOpts.map(([val, lbl]) => `
      <div class="radio-row ${curDev === val ? "selected" : ""}" data-devid="${val}">
        <span class="mono">${curDev === val ? "(*)" : "( )"}</span>
        <span>${esc(lbl)}</span>
      </div>
    `).join("");
    $("devid-radio-stack").querySelectorAll("[data-devid]").forEach((el) => {
      el.addEventListener("click", () => patchConfig({ action: "set_devid", devid: el.dataset.devid }));
    });

    // Features Stack (NVLink, MIG/SYS_PIPE, Media)
    const nvdec = cfg.media?.nvdec || {};
    const nvjpg = cfg.media?.nvjpg_ofa || {};
    const featHtml = [
      `
        <div class="radio-row ${cfg.nvlink_enabled ? "selected" : ""} ${cfg.nvlink_defective ? "disabled" : ""}" id="feat-nvlink">
          <span class="mono">${cfg.nvlink_defective ? "[X]" : (cfg.nvlink_enabled ? "[*]" : "[ ]")}</span>
          <div><strong>NVLink</strong><div style="font-size:11.5px; color:var(--text-secondary);">${cfg.nvlink_defective ? "Defective (locked)" : "Needs U11 bridge mod + FW bypass"}</div></div>
        </div>
      `,
      `
        <div class="radio-row ${cfg.sys_pipe_unlock ? "selected" : ""} ${!cfg.sys_pipe_unlockable ? "disabled" : ""}" id="feat-syspipe">
          <span class="mono">${!cfg.sys_pipe_unlockable ? "[X]" : (cfg.sys_pipe_unlock ? "[*]" : "[ ]")}</span>
          <div><strong>[EXP] MIG / SYS_PIPE</strong><div style="font-size:11.5px; color:var(--text-secondary);">${cfg.sys_pipe_current}/${cfg.sys_pipe_max} healthy pipes</div></div>
        </div>
      `,
      `
        <div class="radio-row ${nvdec.selected ? "selected" : ""} ${!nvdec.available ? "disabled" : ""}" id="feat-nvdec">
          <span class="mono">${!nvdec.available ? "[X]" : (nvdec.selected ? "[*]" : "[ ]")}</span>
          <div><strong>[EXP] NVDEC x5</strong><div style="font-size:11.5px; color:var(--text-secondary);">${esc(!nvdec.currently_disabled ? "Already enabled" : (nvdec.blockers?.length ? nvdec.blockers[0] : nvdec.evidence))}</div></div>
        </div>
      `,
      `
        <div class="radio-row ${nvjpg.selected ? "selected" : ""} ${!nvjpg.available ? "disabled" : ""}" id="feat-nvjpg">
          <span class="mono">${!nvjpg.available ? "[X]" : (nvjpg.selected ? "[*]" : "[ ]")}</span>
          <div><strong>[EXP] NVJPG0 + OFA0</strong><div style="font-size:11.5px; color:var(--text-secondary);">${esc(!nvjpg.currently_disabled ? "Already enabled" : (nvjpg.blockers?.length ? nvjpg.blockers[0] : nvjpg.evidence))}</div></div>
        </div>
      `,
    ];
    $("features-stack").innerHTML = featHtml.join("");
    if (!cfg.nvlink_defective) {
      $("feat-nvlink").addEventListener("click", () => patchConfig({ action: "set_nvlink", enabled: !cfg.nvlink_enabled }));
    }
    if (cfg.sys_pipe_unlockable) {
      $("feat-syspipe").addEventListener("click", () => patchConfig({ action: "set_sys_pipe", enabled: !cfg.sys_pipe_unlock }));
    }
    if (nvdec.available) {
      $("feat-nvdec").addEventListener("click", () => patchConfig({ action: "set_media", which: "nvdec", enabled: !nvdec.selected }));
    }
    if (nvjpg.available) {
      $("feat-nvjpg").addEventListener("click", () => patchConfig({ action: "set_media", which: "nvjpg_ofa", enabled: !nvjpg.selected }));
    }

    // Derived FF Records Table
    const recs = cfg.planned_records || [];
    if (!$("inp-save-plan-path").value) {
      $("inp-save-plan-path").value = cfg.default_save_path || "private/plan.json";
    }
    $("cfg-ff-summary").textContent = recs.length
      ? `1 automatic no-op test row (row ${cfg.next_free_row}) + ${recs.length} real record(s) (rows ${cfg.next_free_row + 1}..${cfg.next_free_row + recs.length}) · ${cfg.free_row_count - (recs.length + 1)} slots remaining after burn.`
      : "No fuse changes selected yet.";

    const cohBox = $("cfg-coherence-box");
    if (cfg.derive_error) {
      cohBox.innerHTML = `<div class="alert danger">${esc(cfg.derive_error)}</div>`;
    } else if (cfg.coherence_issues?.length) {
      cohBox.innerHTML = cfg.coherence_issues.map((i) => `<div class="alert warn">[${esc(i.severity.toUpperCase())}] ${esc(i.message)}</div>`).join("");
    } else if (recs.length) {
      cohBox.innerHTML = `<div class="alert ok">Coherence check: PASS (0 warnings)</div>`;
    } else {
      cohBox.innerHTML = "";
    }

    $("cfg-ff-tbody").innerHTML = recs.length
      ? recs.map((r) => `
          <tr>
            <td class="mono">${r.row}</td>
            <td class="mono"><strong>${esc(r.word)}</strong></td>
            <td class="mono">c${r.chain}/b${r.block}</td>
            <td class="mono">${esc(r.old || "FPF")} → ${esc(r.new)}</td>
            <td class="mono">${esc((r.fields || []).join(", "))}</td>
            <td>${esc(r.note)}</td>
          </tr>
        `).join("")
      : `<tr><td colspan="6" style="color:var(--text-muted);">Toggle Compute, Memory, PCIe, or features above to preview exact FF records.</td></tr>`;
  }

  async function patchConfig(patch) {
    try {
      const cfg = await api("POST", "/api/configure/patch", patch);
      renderConfigState(cfg);
    } catch (e) {
      showToast(e.message, "error");
    }
  }

  async function refreshLoadedPlanReview() {
    if (!state.overview?.plan) {
      $("loaded-plan-review-panel").style.display = "none";
      return;
    }
    try {
      const rev = await api("GET", "/api/plan/review");
      $("loaded-plan-review-panel").style.display = "block";
      $("loaded-plan-meta").textContent = `${rev.path} · ${rev.pci_id} (${rev.sku_name}) · ${rev.record_count} rec(s) · sha=${rev.sha12}`;
      $("loaded-plan-review-log").textContent = (rev.lines || []).join("\n");
    } catch (e) {
      $("loaded-plan-review-panel").style.display = "none";
    }
  }

  // ------------------------------------------------------------------ Step 3: Preflight & Watch
  async function refreshPreflightStatus(full = false) {
    try {
      const st = await api("GET", `/api/status?full=${full ? 1 : 0}`);
      const banner = $("preflight-banner");
      const regs = $("preflight-regs");
      const issuesBox = $("preflight-issues");
      if (!st.available) {
        banner.innerHTML = `<div class="alert warn">${esc(st.detail)}</div>`;
        regs.innerHTML = "";
        issuesBox.innerHTML = "";
        return;
      }
      const cls = st.badge === "GOLDILOCKS" || st.badge === "ARMED" ? "ok" : (st.badge === "POISON" ? "danger" : "warn");
      banner.innerHTML = `<div class="alert ${cls}"><strong>${esc(st.badge)}</strong> — ${esc(st.detail)} (${esc(st.pci_id)} ${esc(st.sku_name)})</div>`;

      const h = st.health;
      const items = [
        ["FUSECTRL", h.fusectrl, h.fusectrl.startsWith("0xe004") ? "ok" : "warn"],
        ["DSOV", String(h.dsov), h.dsov === 0 ? "ok" : "warn"],
        ["CRC", h.crc, h.crc === "0x00000055" ? "ok" : "danger"],
        ["PLM_f8", `${h.plm_f8} (${h.plm_f8_open ? "OPEN" : "CLOSED"})`, h.plm_f8_open ? "ok" : "danger"],
        ["PLM_f4", h.plm_f4 || "n/a", h.plm_f4 === "0xffffffff" ? "ok" : ""],
        ["EFP", h.efp ?? "n/a", h.efp === 1 ? "ok" : ""],
        ["EN_SW", String(h.en_sw), h.en_sw === 1 ? "ok" : ""],
        ["SBFE", h.sbfe ?? "n/a", h.sbfe === 0 ? "ok" : ""],
        ["DEBUGCTRL", h.debugctrl, h.debugctrl === "0x00000000" ? "ok" : "danger"],
      ];
      regs.innerHTML = items.map(([k, v, c]) => `
        <div class="reg-box ${c}">
          <div class="reg-name">${esc(k)}</div>
          <div class="reg-val">${esc(v)}</div>
        </div>
      `).join("");

      const allIssues = [...(st.host_issues || []), ...(st.health_issues || [])];
      issuesBox.innerHTML = allIssues.length
        ? `<div class="alert warn"><strong>Preflight gates not yet met:</strong><ul style="margin:6px 0 0 18px; padding:0;">${allIssues.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>`
        : `<div class="alert ok">All passive preflight &amp; host safety gates PASS.</div>`;

      if (st.badge === "POISON" && state.watchTimer) {
        stopWatch();
        showToast("Live Watch stopped automatically due to 0xBADxxxxx poison read!", "error");
      }
      if (full && st.report_lines?.length) {
        $("preflight-output-panel").style.display = "block";
        $("preflight-output-title").textContent = "Full Passive Status Report";
        $("preflight-output-log").textContent = st.report_lines.join("\n");
      }
    } catch (e) {
      if (state.watchTimer) stopWatch();
      showToast(e.message, "error");
    }
  }

  function stopWatch() {
    if (state.watchTimer) {
      clearInterval(state.watchTimer);
      state.watchTimer = null;
    }
    $("btn-toggle-watch").textContent = "Start Live Watch (1.5s)";
    $("btn-toggle-watch").classList.add("btn-primary");
  }

  function toggleWatch() {
    if (state.watchTimer) {
      stopWatch();
      return;
    }
    $("btn-toggle-watch").textContent = "Stop Live Watch";
    $("btn-toggle-watch").classList.remove("btn-primary");
    refreshPreflightStatus(false);
    state.watchTimer = setInterval(() => {
      if (state.step === "preflight") refreshPreflightStatus(false);
    }, 1500);
  }

  // ------------------------------------------------------------------ Step 4: Burn & SSE
  async function refreshBurnPreview() {
    try {
      const bp = await api("GET", "/api/burn/preview");
      renderBeforeAfterGrid("burn-ba-strip", bp.summary);
      const stateLabel = bp.state === "complete"
        ? `COMPLETE (${bp.committed_count}/${bp.total_count} records already burned)`
        : bp.state === "resume"
          ? `RESUME (${bp.committed_count}/${bp.total_count} burned; ${bp.total_count - bp.committed_count} remaining from row ${bp.next_row})`
          : bp.state === "fresh"
            ? `FRESH (0/${bp.total_count} burned; starts at row ${bp.next_row})`
            : `MISMATCH: ${bp.detail}`;
      $("burn-card-state").textContent = `${stateLabel} · Confirm cadence: ${bp.confirm_mode}`;

      const rw = $("burn-receipt-warn");
      if (bp.needs_receipt) {
        rw.style.display = "block";
        rw.innerHTML = `<div class="alert warn">Resuming real records requires a cold-verified test receipt. Run <strong>Step 5 (Cold-Boot Verify)</strong> first to auto-attach it, or attach one in Step 1.</div>`;
      } else {
        rw.style.display = "none";
      }

      const tbody = $("burn-records-tbody");
      const rowsHtml = [];
      for (const g of bp.groups || []) {
        for (const r of g.records || []) {
          const st = state.burnRowStates[r.row] || "PENDING";
          const stCls = st === "VERIFIED" ? "ok" : (st === "WRITING" ? "info" : (st === "HALT" ? "danger" : ""));
          rowsHtml.push(`
            <tr data-burn-row="${r.row}">
              <td><strong>${esc(g.title)}</strong> <span class="mono" style="color:var(--text-muted);">[${esc(g.token)}]</span></td>
              <td class="mono">${r.row}</td>
              <td class="mono">${esc(r.target)}</td>
              <td class="mono">${esc(r.word)}</td>
              <td>${esc(r.decode)}</td>
              <td><span class="chip ${stCls}" id="burn-row-badge-${r.row}">${esc(st)}</span></td>
            </tr>
          `);
        }
      }
      tbody.innerHTML = rowsHtml.length
        ? rowsHtml.join("")
        : `<tr><td colspan="6" style="color:var(--text-muted);">No remaining records to burn.</td></tr>`;

      $("btn-start-burn").disabled = state.burnRunning || bp.state === "complete" || bp.state === "mismatch";
    } catch (e) {
      $("burn-card-state").textContent = e.message;
      $("burn-records-tbody").innerHTML = "";
    }
  }

  function appendBurnLog(line) {
    const box = $("burn-log");
    if (box.textContent === "No active burn session.") box.textContent = "";
    box.textContent += (box.textContent ? "\n" : "") + line;
    box.scrollTop = box.scrollHeight;
  }

  function setRowBadge(row, text, cls) {
    state.burnRowStates[row] = text;
    const el = $(`burn-row-badge-${row}`);
    if (el) {
      el.textContent = text;
      el.className = "chip " + (cls || "");
    }
  }

  function connectSSE() {
    if (state.sse) state.sse.close();
    const es = new EventSource(`/api/events?token=${encodeURIComponent(TOKEN)}`);
    state.sse = es;
    es.onmessage = (msg) => {
      let ev;
      try { ev = JSON.parse(msg.data); } catch { return; }
      if (ev.type === "log") {
        appendBurnLog(ev.message);
      } else if (ev.type === "record_start") {
        setRowBadge(ev.row, "WRITING", "info");
      } else if (ev.type === "record_verified") {
        setRowBadge(ev.row, "VERIFIED", "ok");
        appendBurnLog(`[OK] Row ${ev.row} verified readback=${ev.readback} CRC=${ev.crc}`);
      } else if (ev.type === "record_halted") {
        setRowBadge(ev.row, "HALT", "danger");
        appendBurnLog(`[HALT] Row ${ev.row}: ${ev.error}`);
      } else if (ev.type === "confirm_required") {
        openConfirmModal(
          "Burn Confirmation Required",
          ev.prompt,
          ev.token,
          async (val, cancelled) => {
            await api("POST", "/api/burn/confirm", {
              confirm_id: ev.confirm_id,
              response: val,
              cancel: cancelled,
            });
          }
        );
      } else if (ev.type === "confirm_resolved") {
        closeConfirmModal();
      } else if (ev.type === "burn_finished") {
        state.burnRunning = false;
        $("btn-abort-burn").style.display = "none";
        $("btn-start-burn").disabled = false;
        closeConfirmModal();
        if (ev.ok) {
          appendBurnLog("=== BURN SESSION COMPLETE (READ-BACK VERIFIED) ===");
          showToast("Burn session completed (read-back verified). Proceed to Cold-Boot Verify.", "ok");
        } else {
          appendBurnLog(`=== BURN STOPPED: ${ev.error} ===`);
          showToast(`Burn stopped: ${ev.error}`, "error");
        }
        refreshOverviewAndCurrent();
      } else if (ev.type === "burn_status") {
        state.burnRunning = !!ev.running;
        $("btn-abort-burn").style.display = ev.running ? "inline-flex" : "none";
        if (ev.pending_confirm) {
          const pc = ev.pending_confirm;
          openConfirmModal("Burn Confirmation Required", pc.prompt, pc.token, async (val, cancelled) => {
            await api("POST", "/api/burn/confirm", { confirm_id: pc.confirm_id, response: val, cancel: cancelled });
          });
        }
      }
    };
  }

  // ------------------------------------------------------------------ Step 5: Cold Verify
  async function refreshColdPreflight() {
    try {
      const pf = await api("GET", "/api/cold/preflight");
      const box = $("cold-preflight-box");
      if (pf.ready) {
        box.innerHTML = `<div class="alert ok"><strong>Cold boot ready!</strong> Auto-detected stage: <strong class="mono">${esc(pf.inferred_stage)}</strong> · Burn boot: <code class="mono">${esc((pf.previous_boot_id || "").slice(0, 12))}</code> → Current boot: <code class="mono">${esc((pf.current_boot_id || "").slice(0, 12))}</code>.${pf.notes?.length ? `<div style="margin-top:4px;">${esc(pf.notes.join(" "))}</div>` : ""}</div>`;
      } else {
        box.innerHTML = `<div class="alert warn"><strong>Not ready for live cold capture yet:</strong><ul style="margin:6px 0 0 18px; padding:0;">${(pf.problems || []).map((p) => `<li>${esc(p)}</li>`).join("")}</ul></div>`;
      }
    } catch (e) {
      $("cold-preflight-box").innerHTML = `<div class="alert warn">${esc(e.message)}</div>`;
    }
  }

  function renderColdResult(res) {
    const banner = $("cold-result-banner");
    banner.style.display = "block";
    if (res.passed) {
      banner.innerHTML = `<div class="alert ok"><strong>PASS (${esc(res.stage.toUpperCase())} STAGE):</strong> Cold snapshot matches all expectations!<div style="margin-top:4px;" class="mono">Snapshot: ${esc(res.snapshot_path)}${res.receipt_path ? ` | Receipt saved &amp; attached: ${esc(res.receipt_path)}` : ""}</div></div>`;
    } else {
      banner.innerHTML = `<div class="alert danger"><strong>FAILED (${esc(res.stage.toUpperCase())} STAGE):</strong> ${esc(res.error)}</div>`;
    }
    const wrap = $("cold-diff-wrap");
    const tbody = $("cold-diff-tbody");
    if (res.comparison?.length) {
      wrap.style.display = "block";
      tbody.innerHTML = res.comparison.map((r) => `
        <tr>
          <td class="mono"><strong>${esc(r.field)}</strong></td>
          <td class="mono">${esc(r.expected)}</td>
          <td class="mono">${esc(r.observed)}</td>
          <td><span class="chip ${r.match ? "ok" : "danger"}">${r.match ? "MATCH" : "DIFF"}</span></td>
        </tr>
      `).join("");
    } else {
      wrap.style.display = "none";
    }
  }

  // ------------------------------------------------------------------ Tools & Overrides
  function renderOverrides(ov) {
    if (!ov) return;
    const defs = [
      ["skip_active_sense", "Skip automatic SENSE_CTRL (passive gates still apply)"],
      ["no_test_receipt", "Allow real stage without cold-verified test receipt"],
      ["mask_mismatch", "Skip initial expected_masks equality check"],
      ["undetermined_ceiling", "Override undetermined/exceeded SM ceiling"],
      ["baseline_mismatch", "Allow baseline FF prefix mismatch / absent"],
      ["unmapped_ff", "Allow unknown/unmapped FF & missing mask"],
      ["allow_manual_debugctrl", "Manual-only: allow DEBUGCTRL target"],
    ];
    const box = $("overrides-list");
    box.innerHTML = defs.map(([k, lbl]) => `
      <label style="display:flex; align-items:center; gap:8px; cursor:pointer;">
        <input type="checkbox" data-ov-key="${k}" ${ov[k] ? "checked" : ""} />
        <span>${esc(lbl)}</span>
      </label>
    `).join("");
    box.querySelectorAll("[data-ov-key]").forEach((cb) => {
      cb.addEventListener("change", async () => {
        try {
          await api("POST", "/api/overrides", { [cb.dataset.ovKey]: cb.checked });
          await refreshOverviewOnly();
        } catch (e) {
          showToast(e.message, "error");
        }
      });
    });
  }

  async function refreshJournal() {
    try {
      const j = await api("GET", "/api/tools/journal?limit=50");
      if (!j.exists) {
        $("journal-log").textContent = "Journal file does not exist yet (will be created on first inspect/burn).";
        return;
      }
      $("journal-log").textContent = (j.recent_events || [])
        .map((ev) => JSON.stringify(ev))
        .join("\n") || "(empty journal)";
    } catch (e) {
      $("journal-log").textContent = e.message;
    }
  }

  // ------------------------------------------------------------------ File Browser Modal
  async function openFileBrowser(title, filter, onPick) {
    state.fileModal.filter = filter || "all";
    state.fileModal.onPick = onPick;
    state.fileModal.selected = null;
    $("file-modal-title").textContent = title;
    $("sel-file-filter").value = state.fileModal.filter;
    $("btn-confirm-file-modal").disabled = true;
    $("btn-file-chmod").style.display = "none";
    $("file-preview").innerHTML = `<div style="color:var(--text-muted);">Select a file on the left to inspect its metadata.</div>`;
    $("modal-files").classList.add("open");
    await loadFileDir(state.fileModal.dir || "private");
  }

  function closeFileBrowser() {
    $("modal-files").classList.remove("open");
  }

  async function loadFileDir(dir) {
    try {
      const data = await api("GET", `/api/files?dir=${encodeURIComponent(dir)}&kind=${encodeURIComponent(state.fileModal.filter)}`);
      state.fileModal.dir = data.directory;
      state.fileModal.selected = null;
      $("file-current-dir").textContent = data.directory;
      $("btn-confirm-file-modal").disabled = true;
      $("btn-file-chmod").style.display = "none";
      const list = $("file-list");
      list.innerHTML = data.entries.map((ent, idx) => {
        const icon = ent.is_dir ? "[DIR]" : (ent.kind === "plan" ? "[PLAN]" : ent.kind === "cold_snapshot" ? "[COLD]" : ent.kind === "test_receipt" ? "[RCPT]" : "[FILE]");
        const badge = ent.sku_match === "match"
          ? `<span class="chip ok">MATCHES GPU</span>`
          : ent.sku_match === "mismatch" || ent.sku_match === "die_mismatch"
            ? `<span class="chip danger">OTHER CARD</span>`
            : (!ent.is_dir && !ent.is_private_0600 ? `<span class="chip warn">${esc(ent.mode_octal)}</span>` : "");
        return `
          <div class="file-row" data-file-idx="${idx}">
            <span class="mono" style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;"><strong>${icon}</strong> ${esc(ent.name)}</span>
            <span>${badge}</span>
          </div>
        `;
      }).join("") || `<div style="padding:12px; color:var(--text-muted);">No matching files in this directory.</div>`;

      list.querySelectorAll("[data-file-idx]").forEach((row) => {
        const ent = data.entries[Number(row.dataset.fileIdx)];
        row.addEventListener("click", () => {
          if (ent.is_dir) {
            loadFileDir(ent.path);
            return;
          }
          list.querySelectorAll(".file-row").forEach((r) => r.classList.remove("selected"));
          row.classList.add("selected");
          state.fileModal.selected = ent;
          $("btn-confirm-file-modal").disabled = ent.is_symlink;
          $("btn-file-chmod").style.display = (!ent.is_private_0600 && !ent.is_symlink) ? "inline-flex" : "none";
          $("file-preview").innerHTML = `
            <div><strong>${esc(ent.name)}</strong></div>
            <div class="mono" style="font-size:11.5px; color:var(--text-secondary); word-break:break-all;">${esc(ent.path)}</div>
            <div>Type: <strong>${esc(ent.summary)}</strong></div>
            <div>Permissions: <span class="mono">${esc(ent.mode_octal)}</span> ${ent.is_private_0600 ? '<span class="chip ok">0600 OK</span>' : '<span class="chip warn">Needs chmod 600</span>'}</div>
            <div>Modified: <span class="mono">${esc(ent.mtime_label)}</span> · Size: ${esc(ent.size_label)}</div>
            ${ent.details?.configured_summary ? `<div class="alert ok" style="font-size:11.5px;">${esc(ent.details.configured_summary)}</div>` : ""}
          `;
        });
        row.addEventListener("dblclick", () => {
          if (!ent.is_dir && !ent.is_symlink && state.fileModal.onPick) {
            closeFileBrowser();
            state.fileModal.onPick(ent.path);
          }
        });
      });
    } catch (e) {
      showToast(e.message, "error");
    }
  }

  // ------------------------------------------------------------------ Confirmation Modal
  function openConfirmModal(title, prompt, expectedToken, onResolve) {
    state.confirmModal.expected = expectedToken;
    state.confirmModal.onResolve = onResolve;
    $("confirm-modal-title").textContent = title;
    $("confirm-modal-prompt").textContent = prompt;
    $("confirm-modal-expected").textContent = expectedToken;
    const inp = $("inp-confirm-phrase");
    inp.value = "";
    $("btn-confirm-submit").disabled = true;
    $("modal-confirm").classList.add("open");
    setTimeout(() => inp.focus(), 30);
  }

  function closeConfirmModal() {
    $("modal-confirm").classList.remove("open");
    state.confirmModal.onResolve = null;
  }

  // ------------------------------------------------------------------ Refresh helpers
  async function refreshOverviewOnly() {
    const ov = await api("GET", "/api/overview");
    renderOverview(ov);
    return ov;
  }

  async function refreshOverviewAndCurrent() {
    await refreshOverviewOnly();
    if (state.step === "burn") await refreshBurnPreview();
    if (state.step === "cold_verify") await refreshColdPreflight();
    if (state.step === "preflight") await refreshPreflightStatus(false);
  }

  // ------------------------------------------------------------------ Wire Events
  function wireEvents() {
    document.querySelectorAll(".step-tab").forEach((btn) => {
      btn.addEventListener("click", () => setStep(btn.dataset.step));
    });

    $("btn-advisor-go").addEventListener("click", () => {
      const target = $("btn-advisor-go").dataset.targetStep;
      if (target) setStep(target);
    });

    $("btn-arm-toggle").addEventListener("click", async () => {
      if (state.overview?.is_simulated) return;
      if (state.overview?.execute_live) {
        await api("POST", "/api/mode/execute", { enabled: false });
        await refreshOverviewAndCurrent();
        showToast("Live Execute turned OFF (safe Read-Only mode).", "ok");
      } else {
        openConfirmModal(
          "Arm Live Hardware Writes",
          "Enabling Live Execute permits physical CMD_PROGRAM writes (still gated by SENSE_CTRL, pre-trigger checks, and group/plan confirmations).",
          "ENABLE LIVE WRITES",
          async (val, cancelled) => {
            closeConfirmModal();
            if (cancelled) return;
            try {
              await api("POST", "/api/mode/execute", { enabled: true, confirm_phrase: val });
              await refreshOverviewAndCurrent();
              showToast("Live Execute ARMED.", "ok");
            } catch (e) {
              showToast(e.message, "error");
            }
          }
        );
      }
    });

    // Step 1
    $("btn-refresh-gpus").addEventListener("click", refreshGpus);
    $("btn-detach-driver").addEventListener("click", async () => {
      try {
        const res = await api("POST", "/api/target/detach", {});
        $("detach-log-wrap").style.display = "block";
        $("detach-log").textContent = (res.logs || []).join("\n") || "Host is clear for BAR0 access.";
        await refreshOverviewOnly();
        await refreshGpus();
        showToast("GPU detached and bound cleanly for BAR0 access.", "ok");
      } catch (e) {
        showToast(e.message, "error");
      }
    });
    $("btn-pick-sim-pcie").addEventListener("click", () => {
      openFileBrowser("Select Sim-PCIe Snapshot (mode 0600 JSON)", "fake_snapshot", async (path) => {
        try {
          const ov = await api("POST", "/api/target/sim", { path, mode: "sim_pcie" });
          renderOverview(ov);
          showToast(`Loaded Sim-PCIe target: ${path}`, "ok");
        } catch (e) { showToast(e.message, "error"); }
      });
    });
    $("btn-pick-fake").addEventListener("click", () => {
      openFileBrowser("Select Fake Snapshot (mode 0600 JSON)", "fake_snapshot", async (path) => {
        try {
          const ov = await api("POST", "/api/target/sim", { path, mode: "fake" });
          renderOverview(ov);
          showToast(`Loaded Fake target: ${path}`, "ok");
        } catch (e) { showToast(e.message, "error"); }
      });
    });
    $("btn-save-journal").addEventListener("click", async () => {
      try {
        const ov = await api("POST", "/api/settings", { journal_path: $("inp-journal-path").value.trim() });
        renderOverview(ov);
        showToast("Journal path updated.", "ok");
      } catch (e) { showToast(e.message, "error"); }
    });
    $("btn-pick-receipt").addEventListener("click", () => {
      openFileBrowser("Select Cold-Verified Test Receipt", "test_receipt", async (path) => {
        try {
          const ov = await api("POST", "/api/settings", { receipt_path: path });
          renderOverview(ov);
          showToast(`Attached test receipt: ${path}`, "ok");
        } catch (e) { showToast(e.message, "error"); }
      });
    });
    $("btn-clear-receipt").addEventListener("click", async () => {
      try {
        const ov = await api("POST", "/api/settings", { receipt_path: null });
        renderOverview(ov);
        showToast("Cleared test receipt.", "ok");
      } catch (e) { showToast(e.message, "error"); }
    });

    // Step 2
    $("btn-read-card-cfg").addEventListener("click", async () => {
      try {
        const cfg = await api("POST", "/api/configure/start", { refresh: true });
        $("inp-save-plan-path").value = cfg.default_save_path || "private/plan.json";
        renderConfigState(cfg);
        showToast("Read card silicon state read-only.", "ok");
      } catch (e) { showToast(e.message, "error"); }
    });
    $("btn-open-plan").addEventListener("click", () => {
      openFileBrowser("Open Saved Burn Plan (mode 0600 JSON)", "plan", async (path) => {
        try {
          const res = await api("POST", "/api/plan/load", { path });
          await refreshOverviewOnly();
          renderBeforeAfterGrid("cfg-ba-grid", res.review.summary);
          $("loaded-plan-review-panel").style.display = "block";
          $("loaded-plan-meta").textContent = `${res.review.path} · ${res.review.pci_id} (${res.review.sku_name}) · ${res.review.record_count} rec(s) · sha=${res.review.sha12}`;
          $("loaded-plan-review-log").textContent = (res.review.lines || []).join("\n");
          showToast(res.note || `Loaded valid plan: ${path}`, "ok");
        } catch (e) { showToast(e.message, "error"); }
      });
    });
    $("tile-all-compute").addEventListener("click", () => patchConfig({ action: "toggle_all_compute" }));
    $("tile-all-memory").addEventListener("click", () => patchConfig({ action: "toggle_all_memory" }));
    $("btn-reset-compute").addEventListener("click", () => patchConfig({ action: "reset_compute" }));
    $("btn-reset-memory").addEventListener("click", () => patchConfig({ action: "reset_memory" }));
    $("btn-cfg-reset-all").addEventListener("click", () => patchConfig({ action: "reset_all" }));
    $("btn-refresh-plan-review").addEventListener("click", refreshLoadedPlanReview);
    $("btn-save-plan").addEventListener("click", async () => {
      const path = $("inp-save-plan-path").value.trim();
      try {
        const res = await api("POST", "/api/configure/save", { path, overwrite: false });
        if (res.collision) {
          openConfirmModal(
            "Overwrite Existing Plan?",
            `${res.message} Type OVERWRITE to replace ${res.path}, or Cancel and pick a different filename.`,
            "OVERWRITE",
            async (val, cancelled) => {
              closeConfirmModal();
              if (cancelled || val.trim() !== "OVERWRITE") return;
              try {
                const res2 = await api("POST", "/api/configure/save", { path, overwrite: true });
                await refreshOverviewOnly();
                await refreshLoadedPlanReview();
                showToast(`Saved & loaded ${res2.path} (${res2.record_count} records).`, "ok");
              } catch (e) { showToast(e.message, "error"); }
            }
          );
          return;
        }
        await refreshOverviewOnly();
        await refreshLoadedPlanReview();
        showToast(`Saved & loaded ${res.path} (${res.record_count} records).`, "ok");
      } catch (e) { showToast(e.message, "error"); }
    });

    // Step 3
    $("btn-toggle-watch").addEventListener("click", toggleWatch);
    $("btn-read-status-once").addEventListener("click", () => refreshPreflightStatus(true));
    $("btn-run-dry-run").addEventListener("click", async () => {
      try {
        const res = await api("POST", "/api/preflight/dry-run", {});
        $("preflight-output-panel").style.display = "block";
        $("preflight-output-title").textContent = "Passive Dry-Run Preview (No Fuses Touched)";
        $("preflight-output-log").textContent = (res.lines || []).join("\n");
        showToast("Passive dry-run complete.", "ok");
      } catch (e) { showToast(e.message, "error"); }
    });
    $("btn-run-sense").addEventListener("click", () => {
      openConfirmModal(
        "Run Active Sense (SENSE_CTRL)",
        "SENSE_CTRL does not burn fuses and restores EFP 0->1 while keeping DSOV=0, but only one SENSE_CTRL is allowed per glitch window and a too-deep frame can collapse.",
        "SENSE",
        async (val, cancelled) => {
          closeConfirmModal();
          if (cancelled) return;
          try {
            const res = await api("POST", "/api/preflight/sense", { confirm_phrase: val });
            await refreshPreflightStatus(false);
            await refreshOverviewOnly();
            showToast(res.passed ? "Active sense PASS — frame armed." : `Active sense issues: ${res.issues.join("; ")}`, res.passed ? "ok" : "error");
          } catch (e) { showToast(e.message, "error"); }
        }
      );
    });

    // Step 4
    $("sel-confirm-mode").addEventListener("change", async () => {
      try {
        const ov = await api("POST", "/api/settings", { confirm_mode: $("sel-confirm-mode").value });
        renderOverview(ov);
        await refreshBurnPreview();
        showToast(`Burn confirmation cadence set to ${$("sel-confirm-mode").value}.`, "ok");
      } catch (e) { showToast(e.message, "error"); }
    });
    $("btn-refresh-burn").addEventListener("click", refreshBurnPreview);
    $("btn-start-burn").addEventListener("click", async () => {
      try {
        state.burnRowStates = {};
        $("burn-log").textContent = "Starting burn session...";
        await api("POST", "/api/burn/start", {});
        state.burnRunning = true;
        $("btn-start-burn").disabled = true;
        $("btn-abort-burn").style.display = "inline-flex";
      } catch (e) {
        showToast(e.message, "error");
      }
    });
    $("btn-abort-burn").addEventListener("click", async () => {
      try {
        await api("POST", "/api/burn/abort", {});
        closeConfirmModal();
        showToast("Sent abort to pending burn prompt.", "ok");
      } catch (e) { showToast(e.message, "error"); }
    });

    // Step 5
    $("btn-cold-preflight").addEventListener("click", refreshColdPreflight);
    $("btn-cold-verify-live").addEventListener("click", async () => {
      try {
        const stage = $("sel-cold-stage").value || null;
        const res = await api("POST", "/api/cold/verify-live", { stage });
        renderColdResult(res);
        await refreshOverviewOnly();
        await refreshColdPreflight();
        showToast(res.passed ? `Cold verification PASS (${res.stage})!` : `Cold verification FAILED: ${res.error}`, res.passed ? "ok" : "error");
      } catch (e) { showToast(e.message, "error"); }
    });
    $("btn-cold-verify-file").addEventListener("click", () => {
      openFileBrowser("Select Cold-Boot Snapshot JSON", "cold_snapshot", async (path) => {
        try {
          const stage = $("sel-cold-stage").value || null;
          const res = await api("POST", "/api/cold/verify-file", { path, stage });
          renderColdResult(res);
          await refreshOverviewOnly();
          showToast(res.passed ? `Cold verification PASS (${res.stage})!` : `Cold verification FAILED: ${res.error}`, res.passed ? "ok" : "error");
        } catch (e) { showToast(e.message, "error"); }
      });
    });

    // Tools
    $("btn-decode-word").addEventListener("click", async () => {
      try {
        const d = await api("GET", `/api/tools/decode?word=${encodeURIComponent($("inp-decode-word").value.trim())}`);
        const lines = [
          `Word  : ${d.word}  (chain=${d.chain}, block=${d.block}, data=${d.data}, IFF=0)`,
          "Fields:",
          ...(d.fields.length ? d.fields.map((f) => `  - ${f.key} (${f.name}) bits ${f.bits} [${f.category}, burnable=${f.burnable}]`) : ["  (none in known map)"]),
        ];
        $("decode-output").textContent = lines.join("\n");
      } catch (e) {
        $("decode-output").textContent = `ERROR: ${e.message}`;
      }
    });
    $("btn-reset-overrides").addEventListener("click", async () => {
      await api("POST", "/api/overrides", { reset: true });
      await refreshOverviewOnly();
      showToast("All overrides reset to OFF.", "ok");
    });
    $("btn-read-reg").addEventListener("click", async () => {
      try {
        const r = await api("GET", `/api/tools/register?target=${encodeURIComponent($("inp-reg-target").value.trim())}`);
        $("tools-inspect-log").textContent = `${r.target} (${r.address}) = ${r.value}${r.poison ? "  [POISON!]" : ""}`;
      } catch (e) { $("tools-inspect-log").textContent = `ERROR: ${e.message}`; }
    });
    $("btn-read-rows").addEventListener("click", async () => {
      try {
        const rows = await api("GET", `/api/tools/rows?start=${$("inp-row-start").value}&end=${$("inp-row-end").value}`);
        $("tools-inspect-log").textContent = rows.map((r) => `row ${String(r.row).padStart(3, " ")}: ${r.word}  ${r.empty ? "[EMPTY]" : r.decoded}`).join("\n");
      } catch (e) { $("tools-inspect-log").textContent = `ERROR: ${e.message}`; }
    });
    $("btn-read-exp").addEventListener("click", async () => {
      try {
        const exp = await api("GET", "/api/tools/experimental");
        $("tools-inspect-log").textContent = (exp.lines || []).join("\n");
      } catch (e) { $("tools-inspect-log").textContent = `ERROR: ${e.message}`; }
    });
    $("btn-refresh-journal").addEventListener("click", refreshJournal);

    // File Modal Controls
    $("btn-close-file-modal").addEventListener("click", closeFileBrowser);
    $("btn-cancel-file-modal").addEventListener("click", closeFileBrowser);
    $("sel-file-filter").addEventListener("change", () => {
      state.fileModal.filter = $("sel-file-filter").value;
      loadFileDir(state.fileModal.dir);
    });
    $("btn-files-private").addEventListener("click", () => loadFileDir("private"));
    $("btn-files-cwd").addEventListener("click", () => loadFileDir("."));
    $("btn-file-chmod").addEventListener("click", async () => {
      if (!state.fileModal.selected) return;
      try {
        await api("POST", "/api/files/chmod600", { path: state.fileModal.selected.path });
        showToast("Fixed file permissions to 0600.", "ok");
        await loadFileDir(state.fileModal.dir);
      } catch (e) { showToast(e.message, "error"); }
    });
    $("btn-confirm-file-modal").addEventListener("click", () => {
      if (state.fileModal.selected && state.fileModal.onPick) {
        const p = state.fileModal.selected.path;
        closeFileBrowser();
        state.fileModal.onPick(p);
      }
    });

    // Confirmation Modal Controls
    $("inp-confirm-phrase").addEventListener("input", () => {
      $("btn-confirm-submit").disabled = $("inp-confirm-phrase").value.trim() !== state.confirmModal.expected;
    });
    $("inp-confirm-phrase").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !$("btn-confirm-submit").disabled) {
        $("btn-confirm-submit").click();
      } else if (e.key === "Escape") {
        $("btn-confirm-cancel").click();
      }
    });
    $("btn-confirm-submit").addEventListener("click", async () => {
      const fn = state.confirmModal.onResolve;
      const val = $("inp-confirm-phrase").value;
      if (fn) await fn(val, false);
    });
    $("btn-confirm-cancel").addEventListener("click", async () => {
      const fn = state.confirmModal.onResolve;
      if (fn) await fn(null, true);
      else closeConfirmModal();
    });
  }

  // ------------------------------------------------------------------ Boot
  async function init() {
    wireEvents();
    connectSSE();
    try {
      const ov = await refreshOverviewOnly();
      await refreshGpus();
      if (ov.recommended_step && ov.recommended_step !== "target") {
        // Keep user on Target initially unless they already have a restored workspace session.
        if (ov.has_target) setStep(ov.recommended_step);
      }
    } catch (e) {
      showToast(`Failed to connect to vfi_burn server: ${e.message}`, "error");
    }
  }

  init();
})();
