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
  const courtPlayers = document.getElementById("court-players");
  const cardTemplate = document.getElementById("player-card-template");
  const SVG_NS = "http://www.w3.org/2000/svg";
  const TEAM_COLORS = ["#39e4f5", "#b693ff"];
  // Per-player DOM (card, foot marker, court dot), rebuilt only when the roster changes.
  let views = new Map();
  let rosterKey = "";
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

  const teamOf = (player) => (player?.team === 1 ? 1 : 0);

  function syncRoster(players) {
    const key = players.map((player) => `${player.id}:${teamOf(player)}`).join(",");
    if (key === rosterKey) return;
    rosterKey = key;
    views.forEach((view) => {
      view.card.remove();
      view.marker.remove();
      view.dot.remove();
    });
    views = new Map();
    players.forEach((player, index) => {
      const team = teamOf(player);
      const card = cardTemplate.content.firstElementChild.cloneNode(true);
      card.querySelector(".player-number").textContent = String(Number(player.id) + 1).padStart(2, "0");
      document.getElementById(`team-cards-${team}`).append(card);
      const marker = document.createElement("div");
      marker.className = "player-marker";
      markerLayer.append(marker);
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("class", "court-player");
      dot.setAttribute("r", "7");
      courtPlayers.append(dot);
      views.set(player.id ?? index, {
        card,
        marker,
        dot,
        name: card.querySelector(".player-name"),
        speed: card.querySelector(".player-speed"),
        distance: card.querySelector(".player-distance"),
      });
    });
  }

  function placeOnCourt(element, point) {
    element.style.display = point ? "block" : "none";
    if (point) element.setAttribute("transform", `translate(${1 + (point[0] / 13.4) * 400} ${1 + (point[1] / 6.1) * 182})`);
  }

  function renderPlayer(player, oldPlayer, amount, view) {
    const imgPoint = interpolatePoint(oldPlayer?.img_xy, player.img_xy, amount);
    const courtPoint = interpolatePoint(oldPlayer?.court_xy_m, player.court_xy_m, amount);
    const isTracked = String(player.tracked === true);
    const playerColor = typeof player.color === "string" && /^#[\da-f]{6}$/i.test(player.color)
      ? player.color
      : TEAM_COLORS[teamOf(player)];

    view.name.textContent = player.name || `PLAYER ${Number(player.id) + 1}`;
    view.speed.textContent = format(interpolateNumber(oldPlayer?.speed_mps, player.speed_mps, amount));
    view.distance.textContent = format(interpolateNumber(oldPlayer?.dist_m, player.dist_m, amount));
    for (const element of [view.card, view.marker, view.dot]) {
      element.dataset.tracked = isTracked;
      element.style.setProperty("--player-color", playerColor);
    }
    if (imgPoint) {
      view.marker.style.display = "block";
      view.marker.style.left = `${Math.max(0, Math.min(100, (imgPoint[0] / FRAME_WIDTH) * 100))}vw`;
      view.marker.style.top = `${Math.max(0, Math.min(100, (imgPoint[1] / FRAME_HEIGHT) * 100))}vh`;
    } else {
      view.marker.style.display = "none";
    }
    placeOnCourt(view.dot, courtPoint);
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
      const mode = targetState.match.mode === "doubles" ? "doubles" : "singles";
      document.querySelector(".hud").dataset.mode = mode;
      document.getElementById("mode-label").textContent = mode.toUpperCase();
      const players = targetState.players.filter((player) => player && typeof player === "object");
      syncRoster(players);
      players.forEach((player, index) => {
        const id = player.id ?? index;
        const oldPlayer = prior.players?.find((candidate) => candidate?.id === player.id);
        renderPlayer(player, oldPlayer, amount, views.get(id));
      });

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
      // Court map shows only the estimated landing spot, never the shuttle in flight (CONTRACTS).
      const landing = shuttle.landing_xy_m_est;
      placeOnCourt(document.getElementById("court-shuttle"), Array.isArray(landing) && landing.length >= 2 ? landing : null);
    }
    window.requestAnimationFrame(render);
  }

  connect();
  window.requestAnimationFrame(render);
})();
