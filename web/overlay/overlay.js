"use strict";

(() => {
  const params = new URLSearchParams(window.location.search);
  const FRAME_WIDTH = Number(params.get("width")) || 1280;
  const FRAME_HEIGHT = Number(params.get("height")) || 720;
  const socketUrl = params.get("ws") || "ws://localhost:8765/";
  const connection = document.getElementById("connection");
  const lostBanner = document.getElementById("signal-lost");
  const markerLayer = document.getElementById("player-markers");
  const shuttleMarker = document.getElementById("shuttle-marker");
  const markers = [0, 1].map((id) => {
    const element = document.createElement("div");
    element.className = `player-marker ${id === 0 ? "cyan" : "violet"}`;
    element.dataset.tracked = "false";
    markerLayer.append(element);
    return element;
  });
  let previousState = null;
  let targetState = null;
  let receivedAt = 0;
  let retryDelay = 500;
  let socket = null;

  const format = (value, digits = 1) => value === null || value === undefined || !Number.isFinite(Number(value))
    ? "—"
    : Number(value).toFixed(digits);

  function setConnection(state, label) {
    connection.dataset.state = state;
    connection.lastElementChild.textContent = label;
    lostBanner.hidden = state !== "lost";
  }

  function connect() {
    setConnection("connecting", "CONNECTING");
    try {
      socket = new WebSocket(socketUrl);
    } catch (_error) {
      setConnection("lost", "OFFLINE");
      scheduleReconnect();
      return;
    }
    socket.addEventListener("open", () => {
      retryDelay = 500;
      setConnection("live", "LIVE");
    });
    socket.addEventListener("message", (event) => {
      try {
        const next = JSON.parse(event.data);
        if (next.v !== 1 || !next.match || !next.rally || !Array.isArray(next.players)) return;
        previousState = targetState;
        targetState = next;
        receivedAt = performance.now();
        setConnection("live", "LIVE");
      } catch (_error) {
        // Ignore malformed frames and keep the last valid state on screen.
      }
    });
    socket.addEventListener("error", () => socket.close());
    socket.addEventListener("close", () => {
      setConnection("lost", "OFFLINE");
      scheduleReconnect();
    });
  }

  function scheduleReconnect() {
    window.setTimeout(connect, retryDelay);
    retryDelay = Math.min(retryDelay * 2, 5000);
  }

  function interpolatePoint(from, to, amount) {
    if (!Array.isArray(to) || to.length < 2) return null;
    if (!Array.isArray(from) || from.length < 2) return to;
    return [from[0] + (to[0] - from[0]) * amount, from[1] + (to[1] - from[1]) * amount];
  }

  function interpolateNumber(from, to, amount) {
    if (to === null || to === undefined || !Number.isFinite(Number(to))) return null;
    if (from === null || from === undefined || !Number.isFinite(Number(from))) return Number(to);
    return Number(from) + (Number(to) - Number(from)) * amount;
  }

  function renderPlayer(player, oldPlayer, amount, index) {
    const card = document.getElementById(`player-card-${index}`);
    const name = document.getElementById(`player-name-${index}`);
    const speed = document.getElementById(`player-speed-${index}`);
    const distance = document.getElementById(`player-distance-${index}`);
    const marker = markers[index];
    const imgPoint = interpolatePoint(oldPlayer?.img_xy, player?.img_xy, amount);
    const courtPoint = interpolatePoint(oldPlayer?.court_xy_m, player?.court_xy_m, amount);
    const speedValue = interpolateNumber(oldPlayer?.speed_mps, player?.speed_mps, amount);
    const distValue = interpolateNumber(oldPlayer?.dist_m, player?.dist_m, amount);
    const isTracked = Boolean(player?.tracked);
    const playerColor = typeof player?.color === "string" && /^#[\da-f]{6}$/i.test(player.color)
      ? player.color
      : index === 0 ? "#39e4f5" : "#b693ff";

    name.textContent = player?.name || `PLAYER ${index + 1}`;
    speed.textContent = format(speedValue);
    distance.textContent = format(distValue);
    card.dataset.tracked = String(isTracked);
    card.style.setProperty("--player-color", playerColor);
    marker.dataset.tracked = String(isTracked);
    marker.style.setProperty("--player-color", playerColor);
    if (imgPoint) {
      marker.style.display = "block";
      marker.style.left = `${Math.max(0, Math.min(100, (imgPoint[0] / FRAME_WIDTH) * 100))}vw`;
      marker.style.top = `${Math.max(0, Math.min(100, (imgPoint[1] / FRAME_HEIGHT) * 100))}vh`;
    } else {
      marker.style.display = "none";
    }
    const courtDot = document.getElementById(`court-player-${index}`);
    courtDot.style.fill = playerColor;
    if (courtPoint) {
      courtDot.style.display = "block";
      courtDot.setAttribute("cx", String(1 + (courtPoint[0] / 13.4) * 400));
      courtDot.setAttribute("cy", String(1 + (courtPoint[1] / 6.1) * 182));
    } else {
      courtDot.style.display = "none";
    }
  }

  function render() {
    if (targetState) {
      const amount = Math.max(0, Math.min(1, (performance.now() - receivedAt) / (1000 / 30)));
      const prior = previousState || targetState;
      const rally = targetState.rally;
      const oldRally = prior.rally || rally;
      const rallyTimeMs = interpolateNumber(oldRally.t_ms, rally.t_ms, amount);
      document.getElementById("rally-time").textContent = format(rallyTimeMs === null ? null : rallyTimeMs / 1000);
      document.getElementById("rally-shots").textContent = rally.shots ?? "—";
      const score = targetState.match.score || [];
      document.getElementById("score-0").textContent = score[0] ?? "—";
      document.getElementById("score-1").textContent = score[1] ?? "—";
      const games = targetState.match.games || [];
      document.getElementById("games-line").textContent = `GAMES ${games[0] ?? "—"} : ${games[1] ?? "—"}`;
      const lastShot = targetState.shuttle?.last_shot;
      document.getElementById("shot-speed").textContent = format(lastShot?.speed_kmh_est, 0);
      targetState.players.slice(0, 2).forEach((player, index) => {
        renderPlayer(player, prior.players?.[index], amount, index);
      });
      for (let index = targetState.players.length; index < 2; index += 1) {
        renderPlayer(null, null, amount, index);
      }

      const shuttle = targetState.shuttle || {};
      const oldShuttle = prior.shuttle || shuttle;
      const shuttleImg = interpolatePoint(oldShuttle.img_xy, shuttle.img_xy, amount);
      shuttleMarker.dataset.status = shuttle.status || "lost";
      document.getElementById("shuttle-status").textContent = `SHUTTLE ${(shuttle.status || "lost").toUpperCase()}`;
      if (shuttleImg && shuttle.status !== "lost") {
        shuttleMarker.style.display = "block";
        shuttleMarker.style.left = `${Math.max(0, Math.min(100, (shuttleImg[0] / FRAME_WIDTH) * 100))}vw`;
        shuttleMarker.style.top = `${Math.max(0, Math.min(100, (shuttleImg[1] / FRAME_HEIGHT) * 100))}vh`;
      } else {
        shuttleMarker.style.display = "none";
      }
      const shuttleCourt = interpolatePoint(oldShuttle.court_xy_m, shuttle.court_xy_m, amount);
      const courtShuttle = document.getElementById("court-shuttle");
      if (shuttleCourt && shuttle.status !== "lost") {
        courtShuttle.style.display = "block";
        courtShuttle.setAttribute("cx", String(1 + (shuttleCourt[0] / 13.4) * 400));
        courtShuttle.setAttribute("cy", String(1 + (shuttleCourt[1] / 6.1) * 182));
      } else {
        courtShuttle.style.display = "none";
      }
    }
    window.requestAnimationFrame(render);
  }

  connect();
  window.requestAnimationFrame(render);
})();
