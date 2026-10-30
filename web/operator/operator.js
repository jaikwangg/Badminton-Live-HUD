"use strict";

(() => {
  const params = new URLSearchParams(window.location.search);
  // The page is served by the operator API, so API calls are same-origin unless ?api= is given.
  const API = params.get("api") || "";
  const SOCKET_URL = params.get("ws") || `ws://${window.location.hostname || "localhost"}:8765/`;
  const COLORS = ["#39e4f5", "#7cf0b4", "#b693ff", "#ff9bd2"];
  // Contract order: clockwise from the far-left corner.
  const CORNER_LABELS = ["มุมซ้ายไกล", "มุมขวาไกล", "มุมขวาใกล้", "มุมซ้ายใกล้"];
  const $ = (selector) => document.querySelector(selector);

  const canvas = $("#calibration");
  const context = canvas.getContext("2d");
  let image = null;
  let corners = [];
  let dragging = -1;
  let pendingPoint = null;
  // Fusion may re-send a suggestion with the same ts to correct it; once answered, ignore those.
  const answeredPoints = new Set();
  let roster = [];
  let rosterKey = "";

  function setStatus(selector, message, failed = false) {
    const element = $(selector);
    element.textContent = message;
    element.dataset.error = String(failed);
  }

  async function post(path, body = {}) {
    const started = performance.now();
    const response = await fetch(`${API}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = Array.isArray(result.detail) ? result.detail.map((item) => item.msg).join(", ") : result.detail;
      throw new Error(detail || `Request failed (${response.status})`);
    }
    return Math.round(performance.now() - started);
  }

  // Runs an API action and reports the round trip so the operator sees it took effect.
  async function act(statusSelector, label, path, body) {
    try {
      const ms = await post(path, body);
      setStatus(statusSelector, `${label} (${ms} ms)`);
      return true;
    } catch (error) {
      setStatus(statusSelector, error.message, true);
      return false;
    }
  }

  // ---------- Match setup ----------
  function playerCount() {
    return $("#mode").value === "doubles" ? 4 : 2;
  }

  function renderPlayers() {
    const count = playerCount();
    const previous = readPlayers();
    const container = $("#players");
    container.replaceChildren();
    for (let id = 0; id < count; id += 1) {
      const team = count === 2 ? id : Math.floor(id / 2);
      const old = previous[id];
      const card = document.createElement("fieldset");
      card.className = "player-card";
      card.innerHTML = `
        <legend>ID ${id}</legend>
        <label class="name">ชื่อ<input data-field="name" maxlength="40" /></label>
        <label>สี<input data-field="color" type="color" /></label>
        <label>ทีม<select data-field="team"><option value="0">ทีมเรา</option><option value="1">คู่แข่ง</option></select></label>
        <label>ฝั่งตอนเริ่ม<select data-field="side"><option value="0">ฝั่ง 0</option><option value="1">ฝั่ง 1</option></select></label>`;
      card.querySelector("[data-field=name]").value = old?.name ?? `Player ${id + 1}`;
      card.querySelector("[data-field=color]").value = old?.color ?? COLORS[id];
      // Team and side reset to the mode's default layout; only name and colour carry over.
      card.querySelector("[data-field=team]").value = String(team);
      card.querySelector("[data-field=side]").value = String(team);
      container.append(card);
    }
  }

  function readPlayers() {
    return Array.from(document.querySelectorAll(".player-card"), (card, id) => ({
      id,
      name: card.querySelector("[data-field=name]").value.trim() || `Player ${id + 1}`,
      color: card.querySelector("[data-field=color]").value,
      team: Number(card.querySelector("[data-field=team]").value),
      side: Number(card.querySelector("[data-field=side]").value),
    }));
  }

  $("#mode").addEventListener("change", renderPlayers);
  $("#save-players").addEventListener("click", () => {
    const players = readPlayers();
    act("#player-status", `บันทึกผู้เล่น ${players.length} คนแล้ว`, "/api/players", { mode: $("#mode").value, players });
  });
  renderPlayers();

  // ---------- Calibration: tap 4 corners in order, then drag to fix ----------
  function updateCalibrationUi() {
    const hint = $("#calibration-hint");
    hint.hidden = !image;
    hint.textContent = corners.length < 4
      ? `แตะจุดที่ ${corners.length + 1}/4: ${CORNER_LABELS[corners.length]}`
      : "ครบ 4 จุดแล้ว ลากจุดเพื่อแก้ หรือกดบันทึก";
    $("#save-calibration").disabled = corners.length !== 4;
    $("#reset-corners").disabled = !image;
  }

  function drawCalibration() {
    if (!image) return;
    context.drawImage(image, 0, 0);
    const scale = canvas.width / canvas.getBoundingClientRect().width || 1;
    context.lineWidth = 2 * scale;
    context.strokeStyle = "#46e0a8";
    context.beginPath();
    corners.forEach(([x, y], index) => (index === 0 ? context.moveTo(x, y) : context.lineTo(x, y)));
    if (corners.length === 4) context.closePath();
    context.stroke();
    const radius = 11 * scale;
    context.font = `bold ${radius}px sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    corners.forEach(([x, y], index) => {
      context.fillStyle = "#f6c85f";
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "#08111d";
      context.fillText(String(index + 1), x, y);
    });
  }

  function pointOnCanvas(event) {
    const rect = canvas.getBoundingClientRect();
    const clamp = (value, max) => Math.round(Math.max(0, Math.min(max, value)));
    return [
      clamp(((event.clientX - rect.left) * canvas.width) / rect.width, canvas.width),
      clamp(((event.clientY - rect.top) * canvas.height) / rect.height, canvas.height),
    ];
  }

  function useCalibrationImage(blob, source) {
    const next = new Image();
    next.onload = () => {
      URL.revokeObjectURL(next.src);
      image = next;
      // Canvas keeps the source resolution so corners_px are in original frame pixels.
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      canvas.style.display = "block";
      corners = [];
      drawCalibration();
      updateCalibrationUi();
      setStatus("#calibration-status", `${source} ${image.naturalWidth}×${image.naturalHeight}`);
    };
    next.onerror = () => setStatus("#calibration-status", "เปิดภาพไม่ได้", true);
    next.src = URL.createObjectURL(blob);
  }

  $("#load-frame").addEventListener("click", async () => {
    try {
      const response = await fetch(`${API}/api/frame.jpg`, { cache: "no-store" });
      if (response.status === 503) {
        setStatus("#calibration-status", "ยังไม่มีภาพจากกล้อง ตรวจว่ากล้องทำงานอยู่ หรือเลือกไฟล์ภาพแทน", true);
        return;
      }
      if (!response.ok) throw new Error(`โหลดภาพกล้องไม่ได้ (${response.status})`);
      useCalibrationImage(await response.blob(), "ภาพกล้อง");
    } catch (error) {
      setStatus("#calibration-status", error.message, true);
    }
  });

  $("#court-image").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) useCalibrationImage(file, "ภาพ");
  });

  canvas.addEventListener("pointerdown", (event) => {
    if (!image) return;
    const [x, y] = pointOnCanvas(event);
    const grab = (24 * canvas.width) / canvas.getBoundingClientRect().width;
    dragging = corners.findIndex(([cx, cy]) => Math.hypot(x - cx, y - cy) <= grab);
    if (dragging < 0 && corners.length < 4) {
      corners.push([x, y]);
      dragging = corners.length - 1;
    }
    if (dragging >= 0) canvas.setPointerCapture(event.pointerId);
    drawCalibration();
    updateCalibrationUi();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (dragging < 0) return;
    corners[dragging] = pointOnCanvas(event);
    drawCalibration();
  });
  for (const type of ["pointerup", "pointercancel"]) canvas.addEventListener(type, () => { dragging = -1; });

  $("#reset-corners").addEventListener("click", () => {
    corners = [];
    drawCalibration();
    updateCalibrationUi();
  });
  $("#save-calibration").addEventListener("click", () => {
    act("#calibration-status", "บันทึกคาลิเบรตแล้ว", "/api/calibrate", { corners_px: corners });
  });

  // ---------- Live control ----------
  document.querySelectorAll("[data-score-team]").forEach((button) => {
    button.addEventListener("click", () => {
      const team = Number(button.dataset.scoreTeam);
      const delta = Number(button.dataset.delta);
      act("#control-status", `${teamName(team)} ${delta > 0 ? "+1" : "−1"}`, "/api/score", { team, delta });
    });
  });

  function swap(a, b) {
    act("#control-status", `สลับ ${playerName(a)} ↔ ${playerName(b)}`, "/api/swap_ids", { a, b });
  }

  $("#swap-ids").addEventListener("click", () => {
    const a = Number($("#swap-a").value);
    const b = Number($("#swap-b").value);
    if (a === b) {
      setStatus("#control-status", "เลือกผู้เล่นสองคนที่ต่างกัน", true);
      return;
    }
    swap(a, b);
  });

  $("#reset-rally").addEventListener("click", async () => {
    if (await act("#control-status", "เริ่มแรลลี่ใหม่แล้ว", "/api/rally/reset")) {
      if (pendingPoint) answeredPoints.add(pendingPoint.ts);
      pendingPoint = null;
      $("#point-suggestion").hidden = true;
    }
  });

  async function decidePoint(accept) {
    if (!pendingPoint) return;
    const label = accept ? "ยืนยันแต้มแล้ว" : "ปฏิเสธแต้มแล้ว";
    // Marked before the request so a correction arriving mid-request can't re-show it.
    answeredPoints.add(pendingPoint.ts);
    // A 422 means the server no longer holds this suggestion (e.g. rally reset), so drop it too.
    await act("#control-status", label, "/api/point/confirm", { event_ts: pendingPoint.ts, accept });
    pendingPoint = null;
    $("#point-suggestion").hidden = true;
  }
  $("#accept-point").addEventListener("click", () => decidePoint(true));
  $("#reject-point").addEventListener("click", () => decidePoint(false));

  function playerName(id) {
    return roster.find((player) => player.id === id)?.name || `ID ${id}`;
  }

  function teamName(team) {
    const names = roster.filter((player) => player.team === team).map((player) => player.name || `ID ${player.id}`);
    return names.length ? names.join(" / ") : `ทีม ${team}`;
  }

  // Rebuilds name-dependent controls only when the roster changes, not on every 30 Hz frame.
  function renderRoster(players) {
    const key = JSON.stringify(players.map((player) => [player.id, player.name, player.team]));
    if (key === rosterKey) return;
    rosterKey = key;
    roster = players.map(({ id, name, team }) => ({ id, name, team }));
    for (const team of [0, 1]) $(`#team-name-${team}`).textContent = teamName(team);
    for (const [index, selector] of ["#swap-a", "#swap-b"].entries()) {
      const select = $(selector);
      const previous = select.value;
      select.replaceChildren(...roster.map((player) => new Option(`${player.id} · ${playerName(player.id)}`, String(player.id))));
      select.value = roster.some((player) => String(player.id) === previous) ? previous : String(roster[index]?.id ?? "");
    }
    // One-click partner swap per doubles team (the usual ID mix-up).
    const swaps = $("#partner-swaps");
    swaps.replaceChildren();
    for (const team of [0, 1]) {
      const members = roster.filter((player) => player.team === team);
      if (members.length !== 2) continue;
      const button = document.createElement("button");
      button.className = "secondary";
      button.textContent = `สลับคู่หู ${playerName(members[0].id)} ↔ ${playerName(members[1].id)}`;
      button.addEventListener("click", () => swap(members[0].id, members[1].id));
      swaps.append(button);
    }
  }

  const format = (value, digits, unit = "") =>
    value === null || value === undefined || !Number.isFinite(Number(value)) ? "—" : `${Number(value).toFixed(digits)}${unit}`;

  function showGameState(state) {
    renderRoster(state.players || []);
    for (const team of [0, 1]) $(`#score-${team}`).textContent = state.match?.score?.[team] ?? 0;
    const health = state.health || {};
    $("#health-fps").textContent = format(health.fps, 1);
    $("#health-p50").textContent = format(health.latency_ms_p50, 0, " ms");
    $("#health-p99").textContent = format(health.latency_ms_p99, 0, " ms");
    $("#health-dropped").textContent = format(health.dropped, 0);
    $("#workers").replaceChildren(
      ...Object.entries(health.workers || {}).map(([name, status]) => {
        const badge = document.createElement("span");
        badge.className = "worker";
        badge.dataset.status = status;
        badge.textContent = `${name}: ${status}`;
        return badge;
      }),
    );
    // Keyed by ts: a later event with the same ts replaces the shown one rather than adding another.
    const suggestion = (state.events || []).findLast(
      (item) => item.kind === "point_suggested" && !answeredPoints.has(item.ts),
    );
    if (suggestion) {
      const team = suggestion.data?.suggested_team ?? suggestion.data?.suggested_side;
      pendingPoint = { ts: suggestion.ts };
      $("#suggestion-text").textContent = team === 0 || team === 1 ? `แต้มให้ ${teamName(team)}?` : "ไม่แน่ใจว่าแต้มของใคร";
      $("#point-suggestion").hidden = false;
    }
  }

  function setConnection(live) {
    $("#connection").dataset.live = String(live);
    $("#connection b").textContent = live ? "เชื่อมต่อแล้ว" : "ขาดการเชื่อมต่อ";
  }

  function connect() {
    const socket = new WebSocket(SOCKET_URL);
    socket.addEventListener("open", () => setConnection(true));
    socket.addEventListener("message", (event) => {
      try {
        showGameState(JSON.parse(event.data));
      } catch (_error) {
        // Keep the last good state on a malformed frame.
      }
    });
    socket.addEventListener("close", () => {
      setConnection(false);
      window.setTimeout(connect, 1000);
    });
    socket.addEventListener("error", () => socket.close());
  }
  connect();
})();
