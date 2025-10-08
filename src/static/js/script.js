// Minimal chess board UI and backend integration
// Build board, render pieces using simple text symbols, handle clicks and moves.

const pieceSymbols = {
  p: '♟',
  r: '♜',
  n: '♞',
  b: '♝',
  q: '♛',
  k: '♚',
  P: '♙',
  R: '♖',
  N: '♘',
  B: '♗',
  Q: '♕',
  K: '♔',
};

let selected = null;
let boardFen = null;
let currentMode = 'pve';
let aiDepth = 2;
let isBoardRotated = false;
let evePollId = null;

function buildBoard() {
  const boardEl = document.querySelector('.board');
  boardEl.innerHTML = '';
  boardEl.classList.add('chessboard');
  // If rotated, build squares in reverse order so visual layout is flipped
  if (!isBoardRotated) {
    for (let rank = 8; rank >= 1; rank--) {
      for (let file = 1; file <= 8; file++) {
        const sq = document.createElement('div');
        const fileChar = String.fromCharCode('a'.charCodeAt(0) + file - 1);
        sq.dataset.square = `${fileChar}${rank}`;
        sq.className = 'square';
        const isLight = (file + rank) % 2 === 0;
        sq.classList.add(isLight ? 'light' : 'dark');
        sq.addEventListener('click', onSquareClick);
        boardEl.appendChild(sq);
      }
    }
  } else {
    // rotated: start from rank 1..8 and file 8..1 so the grid is reversed
    for (let rank = 1; rank <= 8; rank++) {
      for (let file = 8; file >= 1; file--) {
        const sq = document.createElement('div');
        const fileChar = String.fromCharCode('a'.charCodeAt(0) + file - 1);
        sq.dataset.square = `${fileChar}${rank}`;
        sq.className = 'square';
        const isLight = (file + rank) % 2 === 0;
        sq.classList.add(isLight ? 'light' : 'dark');
        sq.addEventListener('click', onSquareClick);
        boardEl.appendChild(sq);
      }
    }
  }
}

function renderFromFEN(fen) {
  boardFen = fen;
  const boardEl = document.querySelector('.board');
  // clear pieces
  boardEl.querySelectorAll('.square').forEach(s => (s.innerHTML = ''));
  const parts = fen.split(' ');
  const rows = parts[0].split('/');

  // compute tile size to scale sprite
  const sampleSquare = document.querySelector('.square');
  const tile = sampleSquare ? sampleSquare.clientWidth : 60; // px
  const spriteCols = 6;
  const spriteRows = 2;
  const spriteUrl = '/static/img/Chess_Pieces_Sprite.svg';

  for (let r = 0; r < 8; r++) {
    const rank = 8 - r;
    let file = 1;
    for (const ch of rows[r]) {
      if (/[0-9]/.test(ch)) {
        file += parseInt(ch, 10);
      } else {
        const fileChar = String.fromCharCode('a'.charCodeAt(0) + file - 1);
        const sq = document.querySelector(`[data-square="${fileChar}${rank}"]`);
        if (sq) {
          // Use individual PNG images from src/static/img, filenames like WP.png, BK.png, etc.
          const color = ch === ch.toUpperCase() ? 'W' : 'B';
          const kind = ch.toLowerCase();
          const letter = kind.toUpperCase(); // p,n,b,r,q,k -> P,N,B,R,Q,K
          const imgName = `${color}${letter}.png`;
          const img = document.createElement('img');
          img.src = `/static/img/${imgName}`;
          img.alt = ch;
          img.draggable = false;
          img.className = 'piece-img';
          // ensure container for piece (to allow overlays/highlight)
          const wrapper = document.createElement('div');
          wrapper.className = 'piece';
          wrapper.appendChild(img);
          sq.appendChild(wrapper);
        }
        file += 1;
      }
    }
  }
}

function onSquareClick(e) {
  const sq = e.currentTarget.dataset.square;
  const clickedEl = e.currentTarget;
  // if nothing selected, try to select this square if it has a friendly piece
  if (!selected) {
    // only allow selecting if it has a piece image and it's the right color
    const pieceImg = clickedEl.querySelector('.piece-img');
    if (!pieceImg) return; // nothing to select
    // determine color by alt (server starts with white to move). We assume player is white.
    const isWhitePiece = pieceImg.alt === pieceImg.alt.toUpperCase();
    // only allow selecting friendly pieces (white) when it's white's turn
    // fetch board turn to be safe
    const turn = boardFen ? boardFen.split(' ')[1] : 'w';
    const whiteToMove = turn === 'w';
    if ((whiteToMove && !isWhitePiece) || (!whiteToMove && isWhitePiece)) return;

    selected = sq;
    clickedEl.classList.add('selected');
    // request legal moves for this square
    fetch(`/api/legal_moves?from=${sq}`)
      .then(r => r.json())
      .then(data => {
        if (data.moves) showLegalTargets(data.moves);
      })
      .catch(() => {});
  } else {
    // if clicking the same square -> deselect
    if (selected === sq) {
      clearSelection();
      return;
    }
    // if clicking another friendly piece, switch selection
    const pieceImg = clickedEl.querySelector('.piece-img');
    if (pieceImg) {
      const isWhitePiece = pieceImg.alt === pieceImg.alt.toUpperCase();
      const turn = boardFen ? boardFen.split(' ')[1] : 'w';
      const whiteToMove = turn === 'w';
      if ((whiteToMove && isWhitePiece) || (!whiteToMove && !isWhitePiece)) {
        // switch selection
        clearSelection();
        selected = sq;
        clickedEl.classList.add('selected');
        fetch(`/api/legal_moves?from=${sq}`)
          .then(r => r.json())
          .then(data => { if (data.moves) showLegalTargets(data.moves); })
          .catch(() => {});
        return;
      }

    }
    // otherwise attempt to move from selected to this square
    const from = selected;
    const to = sq;
    clearSelection();
    const uci = from + to;
    // handle promotion: if a pawn moved to last rank, prompt for promotion piece
    const pieceFrom = document.querySelector(`[data-square="${from}"] .piece-img`);
    const isPawn = pieceFrom && (pieceFrom.alt.toLowerCase() === 'p');
    const destRank = to[1];
    if (isPawn && (destRank === '8' || destRank === '1')) {
      // prompt for promotion piece
      const promo = prompt('Promote to (q,r,b,n) - default queen', 'q') || 'q';
      const p = promo[0].toLowerCase();
      const valid = ['q','r','b','n'].includes(p) ? p : 'q';
      makeMove(uci + valid);
    } else {
      makeMove(uci);
    }
  }
}

function showLegalTargets(uciList) {
  // uciList is array of UCI strings; highlight destination squares
  // clear any existing highlights first
  document.querySelectorAll('.target').forEach(el => el.classList.remove('target'));
  uciList.forEach(u => {
    const dest = u.slice(2,4);
    const el = document.querySelector(`[data-square="${dest}"]`);
    if (el) el.classList.add('target');
  });
}

function setBoardRotation(humanColor) {
  // humanColor: 'w' | 'b' | null
  const prev = isBoardRotated;
  isBoardRotated = humanColor === 'b';
  if (isBoardRotated !== prev) {
    // rebuild board squares and re-render current FEN
    buildBoard();
    if (boardFen) renderFromFEN(boardFen);
  }
}

function clearSelection() {
  selected = null;
  document.querySelectorAll('.selected').forEach(el => el.classList.remove('selected'));
  document.querySelectorAll('.target').forEach(el => el.classList.remove('target'));
}

async function makeMove(uci) {
  try {
    // only request an AI reply when in pve mode
    const wantAIReply = currentMode === 'pve';
    const resp = await fetch('/api/move', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({uci: uci, ai_reply: wantAIReply}),
    });
    const data = await resp.json();
    if (resp.ok) {
      // clear highlights after making a move
      clearSelection();
      document.querySelectorAll('.target').forEach(el => el.classList.remove('target'));
      renderFromFEN(data.fen);
        // check status
        fetch('/api/status').then(r=>r.json()).then(s=>{
          if (s.game_over) {
            alert('Game over: ' + s.result + ' (' + s.reason + ')');
            // re-enable ai-depth
            document.getElementById('ai-depth').disabled = false;
          }
        }).catch(()=>{});
      // if game over, show popup
      if (data.fen) {
        // check with server whether game over
        // server doesn't currently return game_over flag; use client-side fen check
        const parts = data.fen.split(' ');
        // TODO: could request an endpoint returning game_over/result
      }
    } else {
      alert(data.error || 'Move rejected');
    }
  } catch (err) {
    console.error(err);
    alert('Communication error');
  }
}

async function aiMove() {
  const resp = await fetch('/api/ai_move', {method: 'POST'});
  const data = await resp.json();
  if (resp.ok) renderFromFEN(data.fen);
  else alert(data.error || 'AI move failed');
}

async function newGame() {
  const resp = await fetch('/api/new_game', {method: 'POST'});
  const data = await resp.json();
  if (resp.ok) renderFromFEN(data.fen);
}

async function undo() {
  const resp = await fetch('/api/undo', {method: 'POST'});
  const data = await resp.json();
  if (resp.ok) renderFromFEN(data.fen);
  else alert(data.error || 'Undo failed');
}

function wireButtons() {
  document.getElementById('new-game').addEventListener('click', () => {
    const mode = document.getElementById('mode-select').value;
  const depth = parseInt(document.getElementById('ai-depth').value || '2', 10);
  const ai_delay = parseFloat(document.getElementById('ai-delay').value || '0.4');
    aiDepth = depth;
    currentMode = mode;
    // disable depth input while game running
    document.getElementById('ai-depth').disabled = true;
    if (mode === 'eve') {
      // start AI-vs-AI runner and get board fen
      const ai1 = parseInt(document.getElementById('ai1-depth').value || '2', 10);
      const ai2 = parseInt(document.getElementById('ai2-depth').value || '2', 10);
  fetch('/api/eve_start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ai1_depth: ai1, ai2_depth: ai2, ai_delay: ai_delay})})
        .then(r=>r.json()).then(()=>{
          // fetch current fen via legal_moves (returns fen) to render initial board
          fetch('/api/legal_moves?from=a1').then(r=>r.json()).then(d=>{ setBoardRotation(null); renderFromFEN(d.fen); }).catch(()=>{});
          startEvePolling();
        }).catch(()=>{});
    } else {
      fetch('/api/new_game', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({mode: mode, ai_depth: depth, ai_delay: ai_delay})})
        .then(r => r.json())
        .then(d => { setBoardRotation(d.human_color); renderFromFEN(d.fen); });
    }
  });
  // show/hide eve controls
  const modeSelect = document.getElementById('mode-select');
  const eveControls = document.getElementById('eve-controls');
  modeSelect.addEventListener('change', (e)=>{
    // update current mode immediately when user switches selection
    currentMode = e.target.value;
    if (e.target.value === 'eve') {
      eveControls.style.display = 'block';
      document.getElementById('ai-depth-label').style.display = 'none';
    } else {
      eveControls.style.display = 'none';
      document.getElementById('ai-depth-label').style.display = 'block';
      // if leaving eve mode, stop any polling
      stopEvePolling();
    }
  });

  // EVE start/stop
  document.getElementById('eve-start').addEventListener('click', ()=>{
    const ai1 = parseInt(document.getElementById('ai1-depth').value || '2', 10);
    const ai2 = parseInt(document.getElementById('ai2-depth').value || '2', 10);
    fetch('/api/eve_start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ai1_depth: ai1, ai2_depth: ai2, ai_delay: parseFloat(document.getElementById('ai-delay').value || '0.4')})})
      .then(r=>r.json()).then(()=>{
        // fetch current fen via legal_moves to render board
        fetch('/api/legal_moves?from=a1').then(r=>r.json()).then(d=>{ setBoardRotation(null); renderFromFEN(d.fen); }).catch(()=>{});
        startEvePolling();
      });
  });
  document.getElementById('eve-stop').addEventListener('click', ()=>{
    fetch('/api/eve_stop', {method:'POST'}).then(()=>{
      // reload board state
      fetch('/api/legal_moves?from=a1').then(r=>r.json()).then(d=>{ setBoardRotation(null); renderFromFEN(d.fen); }).catch(()=>{});
      stopEvePolling();
    });
  });
  document.getElementById('reset-board').addEventListener('click', () => {
    // reset same as new game but keep settings
    const mode = document.getElementById('mode-select').value;
    const depth = parseInt(document.getElementById('ai-depth').value || '2', 10);
    aiDepth = depth;
    currentMode = mode;
    if (mode === 'eve') {
      const ai1 = parseInt(document.getElementById('ai1-depth').value || '2', 10);
      const ai2 = parseInt(document.getElementById('ai2-depth').value || '2', 10);
      fetch('/api/eve_start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ai1_depth: ai1, ai2_depth: ai2, ai_delay: parseFloat(document.getElementById('ai-delay').value || '0.4')})})
        .then(r=>r.json()).then(()=>{ fetch('/api/legal_moves?from=a1').then(r=>r.json()).then(d=>{ setBoardRotation(null); renderFromFEN(d.fen); }).catch(()=>{}); startEvePolling(); }).catch(()=>{});
    } else {
      fetch('/api/new_game', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({mode: mode, ai_depth: depth})})
        .then(r => r.json())
        .then(d => { setBoardRotation(d.human_color); renderFromFEN(d.fen); });
    }
  });
  document.getElementById('undo-move').addEventListener('click', undo);
  // train button is a link to /train; no extra handler needed
}

function startEvePolling() {
  stopEvePolling();
  evePollId = setInterval(async ()=>{
    try {
      const s = await fetch('/api/status').then(r=>r.json());
      const d = await fetch('/api/legal_moves?from=a1').then(r=>r.json());
      setBoardRotation(d.human_color || null);
      renderFromFEN(d.fen);
      if (s.game_over) {
        stopEvePolling();
        alert('Game over: ' + s.result + ' (' + s.reason + ')');
      }
    } catch (e) {}
  }, 500);
}

function stopEvePolling() {
  if (evePollId) { clearInterval(evePollId); evePollId = null; }
}

window.addEventListener('DOMContentLoaded', () => {
  buildBoard();
  wireButtons();
  // start
  fetch('/api/new_game', {method: 'POST'})
    .then(r => r.json())
    .then(d => { setBoardRotation(d.human_color); renderFromFEN(d.fen); })
    .catch(() => {});
});
