"""Renderizado con pygame: cámara, esqueleto, teclas, partículas y HUD.

Frontend renovado ("aurora neón"): fondo con degradado y nebulosas animadas,
paneles de cristal, teclas con volumen y brillo, notas que caen tipo *guitar
hero* y una **pantalla de inicio gestual** para elegir canción.

Todo el dibujo ocurre en el hilo principal. La API pública que consume
``app.py`` (``draw_background``, ``draw_camera``, ``draw_hands``, ``draw_keys``,
``draw_falling_notes``, ``draw_hud``, ``draw_menu``, ``draw_help``,
``present``, ``key_rect``, ``to_screen``, ``burst_note``) se mantiene estable.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import config
from .hand_tracker import HandFrame
from .utils import clamp

log = logging.getLogger("mano_musical.render")

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None  # type: ignore[assignment]

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Utilidades de fuentes y color
# ---------------------------------------------------------------------------
def _font(size: int, bold: bool = False, family: Optional[str] = None) -> "pygame.font.Font":
    family = family or config.FONT_UI
    for name in (family, "Segoe UI", "Arial"):
        try:
            return pygame.font.SysFont(name, size, bold=bold)
        except Exception:  # pragma: no cover
            continue
    return pygame.font.Font(None, size)  # pragma: no cover


def _mono(size: int, bold: bool = False) -> "pygame.font.Font":
    try:
        return pygame.font.SysFont(config.FONT_MONO, size, bold=bold)
    except Exception:  # pragma: no cover
        return pygame.font.Font(None, size)


def _lerp_color(a, b, t: float) -> tuple:
    t = clamp(t, 0.0, 1.0)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _scale_color(color, factor: float) -> tuple:
    return tuple(int(clamp(c * factor, 0, 255)) for c in color[:3])


def _vertical_gradient(w: int, h: int, top, bottom) -> "pygame.Surface":
    surf = pygame.Surface((max(1, w), max(1, h)))
    for y in range(h):
        pygame.draw.line(surf, _lerp_color(top, bottom, y / max(1, h - 1)), (0, y), (w, y))
    return surf


def _rounded_mask(w: int, h: int, radius: int) -> "pygame.Surface":
    mask = pygame.Surface((max(1, w), max(1, h)), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h), border_radius=int(radius))
    return mask


def _round_surface(surf: "pygame.Surface", radius: int) -> "pygame.Surface":
    w, h = surf.get_size()
    base = pygame.Surface((w, h), pygame.SRCALPHA)
    base.blit(surf, (0, 0))
    base.blit(_rounded_mask(w, h, radius), (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return base


# ---------------------------------------------------------------------------
# Partículas
# ---------------------------------------------------------------------------
@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple
    size: float
    gravity: float = 260.0

    def update(self, dt: float) -> bool:
        self.life -= dt
        self.vy += self.gravity * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        return self.life > 0


class ParticleSystem:
    """Partículas que brotan al tocar cada nota."""

    def __init__(self, limit: int = 700) -> None:
        self.items: list[Particle] = []
        self.limit = limit

    def burst(self, x: float, y: float, color: tuple, count: int = 22, power: float = 1.0) -> None:
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(60, 360) * power
            life = random.uniform(0.45, 1.15)
            self.items.append(
                Particle(
                    x,
                    y,
                    math.cos(angle) * speed,
                    math.sin(angle) * speed - 90,
                    life,
                    life,
                    color,
                    random.uniform(2.5, 7.0),
                )
            )
        if len(self.items) > self.limit:
            self.items = self.items[-self.limit:]

    def update(self, dt: float) -> None:
        self.items = [p for p in self.items if p.update(dt)]

    def draw(self, surface: "pygame.Surface") -> None:
        for p in self.items:
            ratio = clamp(p.life / p.max_life, 0.0, 1.0)
            radius = max(1, int(p.size * (0.4 + 0.6 * ratio)))
            glow = _scale_color(p.color, 0.5 + 0.5 * ratio)
            pygame.draw.circle(surface, _scale_color(glow, 0.45), (int(p.x), int(p.y)), radius + 2)
            pygame.draw.circle(surface, glow, (int(p.x), int(p.y)), radius)


# ---------------------------------------------------------------------------
# Datos de entrada para el render
# ---------------------------------------------------------------------------
@dataclass
class MenuSong:
    """Tarjeta de canción para la pantalla de inicio."""

    title: str
    bpm: int
    tag: str
    notes: list[str]
    color: tuple


@dataclass
class MenuData:
    """Todo lo que la pantalla de menú necesita dibujar."""

    songs: list[MenuSong] = field(default_factory=list)
    selected: int = 0
    diffs: list[str] = field(default_factory=list)
    diff_index: int = 0
    hands: int = 0
    camera: bool = False
    tracker: bool = False
    fist_progress: float = 0.0
    fist_ready: bool = False
    message: str = ""
    message_timer: float = 0.0
    subtitle: str = ""


@dataclass
class HudData:
    """Datos que alimentan el HUD desde la app."""

    mode: str = "free"
    fps: float = 0.0
    infer_ms: float = 0.0
    hands: int = 0
    active_notes: list[str] = field(default_factory=list)
    camera: bool = True
    tracker: bool = True
    timbre: str = "piano"
    octave: int = 0
    recording: bool = False
    message: str = ""
    submessage: str = ""
    show_help: bool = False
    countdown: float = 0.0
    banner: str = ""
    banner_timer: float = 0.0
    game: Optional[dict] = None
    detection: str = "curl"
    mirror: bool = True
    rotate: bool = False
    swapped: bool = False
    calibrating: bool = False
    calibration: float = 0.0
    sensitivity: float = config.PRESS_ANGLE_ENTER
    fist_progress: float = 0.0


class Renderer:
    """Dibuja un frame completo de Mano Musical."""

    KEY_BAND_HEIGHT = 204
    HUD_HEIGHT = 108

    def __init__(self, width: int = config.WINDOW_WIDTH, height: int = config.WINDOW_HEIGHT) -> None:
        if pygame is None:  # pragma: no cover
            raise RuntimeError("pygame no está disponible")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption(config.WINDOW_TITLE)

        self.font_tiny = _font(13)
        self.font_sm = _font(15)
        self.font_md = _font(21, bold=True)
        self.font_lg = _font(32, bold=True)
        self.font_xl = _font(84, bold=True)
        self.font_title = _font(56, bold=True)
        self.font_sub = _font(20)

        self.particles = ParticleSystem()
        self.view_rect = pygame.Rect(0, self.HUD_HEIGHT, width, height - self.HUD_HEIGHT - self.KEY_BAND_HEIGHT)

        self._text_cache: dict = {}
        self._glow_cache: dict = {}
        self._panel_cache: dict = {}
        self._key_cache: dict = {}
        self._card_cache: dict = {}
        self._bg = None
        self._bg_key = None
        self._scrim_cache: dict = {}
        self._stars: list = []

        # Mapa de la cámara para traducir landmarks -> pantalla.
        self._cam_stage = pygame.Rect(0, 0, 1, 1)
        self._cam_native = None
        self._cam_scale = 1.0
        self._cam_off = (0.0, 0.0)

        self.time = 0.0

    # -------------------------------------------------------------- utilities
    def _text(self, text: str, font, color) -> "pygame.Surface":
        key = (text, id(font), color)
        surf = self._text_cache.get(key)
        if surf is None:
            surf = font.render(text, True, color)
            if len(self._text_cache) > 1200:
                self._text_cache.clear()
            self._text_cache[key] = surf
        return surf

    def _blit_text(self, text: str, font, color, x: int, y: int, center: bool = False) -> pygame.Rect:
        surf = self._text(text, font, color)
        rect = surf.get_rect()
        if center:
            rect.center = (x, y)
        else:
            rect.topleft = (x, y)
        self.screen.blit(surf, rect)
        return rect

    def _wrap(self, text: str, font, max_w: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if font.size(trial)[0] <= max_w or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _glow(self, radius: int, color) -> "pygame.Surface":
        radius = max(8, int(radius))
        key = (radius, color)
        surf = self._glow_cache.get(key)
        if surf is None:
            size = radius * 2
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            steps = max(10, radius // 3)
            for i in range(steps):
                r = int(radius * (steps - i) / steps)
                a = int(95 * ((i + 1) / steps) ** 2)
                pygame.draw.circle(surf, (color[0], color[1], color[2], a), (radius, radius), r)
            if len(self._glow_cache) > 60:
                self._glow_cache.clear()
            self._glow_cache[key] = surf
        return surf

    def _panel(self, w: int, h: int, radius: int = 18, color=config.COLOR_GLASS,
               edge=config.COLOR_GLASS_EDGE, alpha: int = 180) -> "pygame.Surface":
        key = (w, h, radius, color, edge, alpha)
        surf = self._panel_cache.get(key)
        if surf is None:
            surf = pygame.Surface((max(1, w), max(1, h)), pygame.SRCALPHA)
            pygame.draw.rect(surf, (color[0], color[1], color[2], alpha), (0, 0, w, h), border_radius=radius)
            sheen = pygame.Surface((max(1, w), max(1, h)), pygame.SRCALPHA)
            pygame.draw.rect(sheen, (255, 255, 255, 24), (0, 0, w, h // 2), border_radius=radius)
            surf.blit(sheen, (0, 0))
            if edge:
                pygame.draw.rect(surf, (edge[0], edge[1], edge[2], 140), (0, 0, w, h), 2, border_radius=radius)
            if len(self._panel_cache) > 60:
                self._panel_cache.clear()
            self._panel_cache[key] = surf
        return surf

    def _scrim(self, alpha_top: int, alpha_bottom: int) -> "pygame.Surface":
        key = (self.width, self.height, alpha_top, alpha_bottom)
        surf = self._scrim_cache.get(key)
        if surf is None:
            surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            for y in range(self.height):
                a = int(alpha_top + (alpha_bottom - alpha_top) * y / max(1, self.height - 1))
                pygame.draw.line(surf, (4, 6, 16, a), (0, y), (self.width, y))
            if len(self._scrim_cache) > 12:
                self._scrim_cache.clear()
            self._scrim_cache[key] = surf
        return surf

    def _glow_rect(self, rect: pygame.Rect, color, layers: int = 4, radius: int = 18, spread: int = 5) -> None:
        for i in range(layers, 0, -1):
            a = int(64 / (i * i))
            grown = rect.inflate(i * spread * 2, i * spread * 2)
            surf = pygame.Surface(grown.size, pygame.SRCALPHA)
            pygame.draw.rect(surf, (color[0], color[1], color[2], a), (0, 0, grown.width, grown.height),
                             3, border_radius=radius + i * spread)
            self.screen.blit(surf, grown.topleft)

    def _stage_rect(self) -> pygame.Rect:
        top = self.HUD_HEIGHT
        bottom = self.height - self.KEY_BAND_HEIGHT
        return pygame.Rect(0, top, self.width, max(1, bottom - top))

    def to_screen(self, nx: float, ny: float) -> tuple[float, float]:
        if self._cam_native is None:
            r = self.view_rect
            return float(r.x) + float(nx) * float(r.width), float(r.y) + float(ny) * float(r.height)
        w, h = self._cam_native
        return (
            self._cam_stage.x + self._cam_off[0] + float(nx) * w * self._cam_scale,
            self._cam_stage.y + self._cam_off[1] + float(ny) * h * self._cam_scale,
        )

    def key_rect(self, note: str) -> pygame.Rect:
        idx = config.VISUAL_KEYS.index(note)
        n = len(config.VISUAL_KEYS)
        band_top = self.height - self.KEY_BAND_HEIGHT
        pad = 7
        w = self.width / n
        return pygame.Rect(int(idx * w) + pad, band_top + pad, int(w) - 2 * pad, self.KEY_BAND_HEIGHT - 2 * pad)

    def burst_note(self, note: str, screen_pos: tuple[float, float], velocity: float = 1.0) -> None:
        color = config.NOTE_COLORS.get(note, config.COLOR_ACCENT)
        self.particles.burst(screen_pos[0], screen_pos[1], color, count=22, power=0.6 + velocity)

    # ------------------------------------------------------------- background
    def _ensure_bg(self) -> None:
        key = (self.width, self.height)
        if self._bg_key == key and self._bg is not None:
            return
        self._bg = _vertical_gradient(self.width, self.height, config.COLOR_BG_TOP, config.COLOR_BG_BOTTOM)
        random.seed(7)
        self._stars = [
            (random.random() * self.width, random.random() * self.height,
             random.uniform(0.6, 1.8), random.random() * math.tau)
            for _ in range(110)
        ]
        self._bg_key = key
        self._panel_cache.clear()
        self._key_cache.clear()
        self._card_cache.clear()

    def draw_background(self) -> None:
        self._ensure_bg()
        self.screen.blit(self._bg, (0, 0))
        t = self.time
        blobs = [
            (0.18, 0.24, config.COLOR_PURPLE, 340, 0.0),
            (0.86, 0.20, config.COLOR_ACCENT, 280, 1.6),
            (0.62, 0.92, config.COLOR_ACCENT_2, 320, 3.1),
        ]
        for fx, fy, color, rad, phase in blobs:
            x = fx * self.width + math.sin(t * 0.14 + phase) * 90
            y = fy * self.height + math.cos(t * 0.11 + phase) * 70
            glow = self._glow(rad, color)
            glow.set_alpha(int(70 + 40 * (0.5 + 0.5 * math.sin(t * 0.4 + phase))))
            self.screen.blit(glow, (x - glow.get_width() / 2, y - glow.get_height() / 2),
                             special_flags=pygame.BLEND_ADD)
        for sx, sy, size, phase in self._stars:
            b = 0.25 + 0.5 * (0.5 + 0.5 * math.sin(t * 1.7 + phase))
            c = int(170 * b)
            pygame.draw.circle(self.screen, (c, c, min(255, c + 40)), (int(sx), int(sy)), max(1, int(size)))

    # ----------------------------------------------------------------- camera
    def _draw_idle_stage(self, stage: pygame.Rect) -> None:
        self._cam_native = None
        self.view_rect = stage
        t = self.time
        for i in range(0, stage.width, 46):
            alpha = int(20 + 16 * math.sin(t * 1.4 + i * 0.02))
            pygame.draw.line(self.screen, (alpha, min(255, alpha + 24), 70),
                             (stage.x + i, stage.y), (stage.x + i, stage.bottom))
        panel = self._panel(min(760, stage.width - 60), 190, radius=24)
        pr = panel.get_rect(center=stage.center)
        self.screen.blit(panel, pr.topleft)
        self._blit_text("SIN CÁMARA", self.font_title, config.COLOR_TEXT_DIM,
                        stage.centerx, stage.centery - 44, center=True)
        self._blit_text("Usa el teclado ·  A S D F G  H J K L ;",
                        self.font_md, config.COLOR_ACCENT, stage.centerx, stage.centery + 26, center=True)
        self._blit_text("Do Re Mi Fa Sol  La Si Do' Re' Mi'",
                        self.font_sm, config.COLOR_TEXT_DIM, stage.centerx, stage.centery + 62, center=True)

    def draw_camera(self, frame: Optional[np.ndarray], rect: Optional[pygame.Rect] = None,
                    radius: int = 20, dim: int = 34, fill: bool = False) -> None:
        """Dibuja la cámara (recortada con esquinas redondeadas) en ``rect``."""
        stage = rect or self._stage_rect()
        self._cam_stage = stage
        if frame is None or cv2 is None:
            self._draw_idle_stage(stage)
            return
        h, w = frame.shape[:2]
        scale = max(stage.width / w, stage.height / h) if fill else min(stage.width / w, stage.height / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        self._cam_native = (w, h)
        self._cam_scale = scale
        self._cam_off = ((stage.width - nw) / 2.0, (stage.height - nh) / 2.0)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        base = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
        scaled = pygame.transform.smoothscale(base, (nw, nh))

        if fill:
            canvas = pygame.Surface((stage.width, stage.height), pygame.SRCALPHA)
            canvas.blit(scaled, (int(self._cam_off[0]), int(self._cam_off[1])))
            display = _round_surface(canvas, radius)
            pos = stage.topleft
            view = stage
        else:
            display = _round_surface(scaled, radius)
            pos = (stage.x + int(self._cam_off[0]), stage.y + int(self._cam_off[1]))
            view = pygame.Rect(pos, (nw, nh))

        self.view_rect = view
        self.screen.blit(display, pos)
        if dim > 0:
            shade = pygame.Surface(display.get_size(), pygame.SRCALPHA)
            shade.fill((5, 8, 20, dim))
            self.screen.blit(_round_surface(shade, radius), pos)
        pygame.draw.rect(self.screen, config.COLOR_GLASS_EDGE, view, 2, border_radius=radius)

    # ------------------------------------------------------------------ manos
    def draw_hands(self, hand_frame: HandFrame, readings: dict) -> None:
        if not hand_frame.hands:
            return
        for hand in hand_frame.hands:
            label = "IZQ" if hand.handedness == "Left" else "DER"
            pts = [self.to_screen(hand.landmarks[i, 0], hand.landmarks[i, 1]) for i in range(21)]
            for a, b in config.HAND_CONNECTIONS:
                pressed_link = self._link_pressed(hand, readings, a, b)
                color = config.COLOR_ACCENT if pressed_link else config.COLOR_GLASS_EDGE
                pygame.draw.line(self.screen, _scale_color(color, 0.4), pts[a], pts[b], 7)
                pygame.draw.line(self.screen, color, pts[a], pts[b], 3 if pressed_link else 2)
            for i, (sx, sy) in enumerate(pts):
                pygame.draw.circle(self.screen, (238, 242, 255), (int(sx), int(sy)), 3)
            for finger in config.FINGER_ORDER:
                note = config.NOTE_MAP.get((hand.handedness, finger))
                if note is None:
                    continue
                tip_idx = config.FINGER_JOINTS[finger][-1]
                sx, sy = pts[tip_idx]
                reading = readings.get(note, {})
                pressed = bool(reading.get("pressed"))
                color = config.NOTE_COLORS.get(note, config.COLOR_ACCENT)
                radius = 15 if pressed else 10
                if pressed:
                    glow = self._glow(40, color)
                    self.screen.blit(glow, (sx - 40, sy - 40), special_flags=pygame.BLEND_ADD)
                pygame.draw.circle(self.screen, _scale_color(color, 0.35), (int(sx), int(sy)), radius + 6)
                pygame.draw.circle(self.screen, color, (int(sx), int(sy)), radius)
                if pressed:
                    self._blit_text(note, self.font_tiny, (5, 8, 20), int(sx), int(sy), center=True)
            cx, cy = hand.center()
            sx, sy = self.to_screen(cx, cy)
            tag = self._panel(52, 26, radius=13, alpha=150)
            tr = tag.get_rect(center=(int(sx), int(sy) - 34))
            self.screen.blit(tag, tr.topleft)
            self._blit_text(label, self.font_tiny, config.COLOR_TEXT, int(sx), int(sy) - 34, center=True)

    def _link_pressed(self, hand, readings: dict, a: int, b: int) -> bool:
        for finger, joints in config.FINGER_JOINTS.items():
            if a in joints and b in joints:
                note = config.NOTE_MAP.get((hand.handedness, finger))
                return bool(readings.get(note, {}).get("pressed")) if note else False
        return False

    # ----------------------------------------------------------------- teclas
    def _key_surface(self, note: str, active: bool, w: int, h: int) -> "pygame.Surface":
        key = ("key", note, active, w, h)
        surf = self._key_cache.get(key)
        if surf is not None:
            return surf
        color = config.NOTE_COLORS.get(note, config.COLOR_ACCENT)
        if active:
            top = _lerp_color(color, (255, 255, 255), 0.35)
            bottom = color
        else:
            top = _lerp_color(color, config.COLOR_PANEL, 0.70)
            bottom = _lerp_color(color, config.COLOR_PANEL, 0.88)
        surf = _round_surface(_vertical_gradient(w, h, top, bottom), 14)
        sheen = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(sheen, (255, 255, 255, 46), (0, 0, w, int(h * 0.42)), border_radius=14)
        surf.blit(sheen, (0, 0))
        edge = color if active else _lerp_color(color, (0, 0, 0), 0.25)
        pygame.draw.rect(surf, edge, (0, 0, w, h), 3, border_radius=14)
        self._key_cache[key] = surf
        return surf

    def draw_keys(self, active_notes: list[str], flash: Optional[dict] = None) -> None:
        flash = flash or {}
        band_top = self.height - self.KEY_BAND_HEIGHT
        band = pygame.Surface((self.width, self.KEY_BAND_HEIGHT), pygame.SRCALPHA)
        band.fill((8, 10, 26, 170))
        self.screen.blit(band, (0, band_top))
        for note in config.VISUAL_KEYS:
            rect = self.key_rect(note)
            active = note in active_notes
            color = config.NOTE_COLORS.get(note, config.COLOR_ACCENT)
            if active:
                self._glow_rect(rect, color, layers=3, radius=14, spread=4)
            self.screen.blit(self._key_surface(note, active, rect.width, rect.height), rect.topleft)
            solfeo = config.SOLFEGE.get(note[0], "?")
            text_color = (8, 10, 20) if active else config.COLOR_TEXT
            self._blit_text(solfeo, self.font_lg, text_color, rect.centerx, rect.y + rect.height // 2 - 14, center=True)
            self._blit_text(note, self.font_tiny, _scale_color(text_color, 0.75), rect.centerx, rect.y + rect.height // 2 + 22, center=True)
            key_char = next((k for k, v in config.KEYBOARD_MAP.items() if v == note), "")
            cap = self._panel(38, 30, radius=9, color=config.COLOR_PANEL, edge=None,
                              alpha=210 if not active else 90)
            cr = cap.get_rect(center=(rect.centerx, rect.bottom - 26))
            self.screen.blit(cap, cr.topleft)
            self._blit_text(key_char.upper(), self.font_sm, text_color, rect.centerx, rect.bottom - 26, center=True)
        self._blit_text("AIR-PIANO · un dedo, una nota", self.font_sm, config.COLOR_TEXT_DIM, 14, band_top + 6)

    # --------------------------------------------------------- notas que caen
    def draw_falling_notes(self, upcoming: list, horizon: float = config.GAME_FALL_HORIZON) -> None:
        """Notas estilo *guitar hero*: carriles verticales + ladrillos que caen."""
        stage = self._stage_rect()
        hit_y = self.height - self.KEY_BAND_HEIGHT
        travel = max(1, hit_y - stage.y)
        pps = travel / horizon
        for note in config.VISUAL_KEYS:
            r = self.key_rect(note)
            color = config.NOTE_COLORS.get(note, config.COLOR_ACCENT)
            lane = pygame.Surface((r.width, travel), pygame.SRCALPHA)
            lane.fill((*color, 16))
            self.screen.blit(lane, (r.x, stage.y))
            pygame.draw.line(self.screen, (*color, 255), (r.centerx, stage.y), (r.centerx, hit_y), 1)
        for target, delta in upcoming:
            r = self.key_rect(target.note)
            color = config.NOTE_COLORS.get(target.note, config.COLOR_ACCENT)
            bottom = hit_y - delta * pps
            if bottom < stage.y - 40:
                continue
            top = max(stage.y, bottom - max(30.0, target.seconds * pps * 0.92))
            tile = pygame.Rect(r.x + 8, int(top), r.width - 16, max(20, int(bottom - top)))
            surf = _round_surface(_vertical_gradient(tile.width, tile.height, _lerp_color(color, (255, 255, 255), 0.3), color), 10)
            pygame.draw.rect(surf, (255, 255, 255), (0, 0, tile.width, tile.height), 2, border_radius=10)
            if abs(delta) < 0.14:
                self._glow_rect(tile, color, layers=3, radius=10, spread=4)
            self.screen.blit(surf, tile.topleft)
            if tile.height >= 24:
                self._blit_text(target.note, self.font_tiny, (8, 10, 20), tile.centerx, tile.centery, center=True)
        line = _vertical_gradient(stage.width, 6, config.COLOR_ACCENT, config.COLOR_ACCENT_2)
        self.screen.blit(line, (0, hit_y - 1))

    # -------------------------------------------------------------------- HUD
    def _chip(self, x: int, y: int, text: str, color, font=None) -> int:
        font = font or self.font_sm
        tw, th = font.size(text)
        w, h = tw + 24, th + 12
        surf = self._panel(w, h, radius=h // 2, color=config.COLOR_PANEL, edge=color, alpha=150)
        self.screen.blit(surf, (x, y))
        self._blit_text(text, font, color, x + w // 2, y + h // 2, center=True)
        return x + w + 8

    def draw_hud(self, hud: HudData) -> None:
        bar = pygame.Surface((self.width, self.HUD_HEIGHT), pygame.SRCALPHA)
        bar.fill((8, 10, 26, 205))
        self.screen.blit(bar, (0, 0))
        pygame.draw.line(self.screen, config.COLOR_GLASS_EDGE, (0, self.HUD_HEIGHT - 2), (self.width, self.HUD_HEIGHT - 2), 2)

        self._blit_text("MANO", self.font_md, config.COLOR_TEXT, 18, 14)
        self._blit_text("MUSICAL", self.font_md, config.COLOR_ACCENT, 92, 14)
        mode_label = {"free": "LIBRE", "game": "MODO JUEGO", "demo": "DEMO", "replay": "REPLAY", "menu": "MENÚ"}.get(
            hud.mode, hud.mode.upper()
        )
        self._chip(220, 12, mode_label, config.COLOR_ACCENT_2)
        x = self._chip(340, 44, f"{hud.fps:4.0f} FPS", config.COLOR_WARN)
        x = self._chip(x, 44, f"infer {hud.infer_ms:4.1f} ms", config.COLOR_TEXT_DIM)
        x = self._chip(x, 44, f"manos {hud.hands}", config.COLOR_ACCENT)
        x = self._chip(x, 44, f"timbre {hud.timbre}", config.COLOR_PURPLE)
        self._chip(x, 44, f"oct {hud.octave:+d}", config.COLOR_TEXT_DIM)

        notes = " ".join(hud.active_notes) if hud.active_notes else "—"
        self._blit_text(f"Sonando: {notes}", self.font_sm, config.COLOR_TEXT, self.width - 330, 14)
        estado = ("cámara OK" if hud.camera else "sin cámara") + (" · tracker OK" if hud.tracker else " · tracker off")
        color = config.COLOR_OK if (hud.camera and hud.tracker) else config.COLOR_WARN
        self._blit_text(estado, self.font_sm, color, self.width - 300, 44)
        self._blit_text(f"det {hud.detection} · sens {hud.sensitivity:.0f}° · {hud.timbre}",
                        self.font_tiny, config.COLOR_TEXT_DIM, self.width - 300, 72)
        if hud.recording:
            pygame.draw.circle(self.screen, config.COLOR_ACCENT_2, (self.width - 22, 26), 9)
            self._blit_text("REC", self.font_sm, config.COLOR_ACCENT_2, self.width - 44, 18)

        if hud.game:
            self._draw_game_hud(hud.game)
        if hud.countdown > 0:
            self._blit_text(str(int(math.ceil(hud.countdown))), self.font_xl, config.COLOR_ACCENT,
                            self.width // 2, self.height // 2, center=True)
        if hud.banner and hud.banner_timer > 0:
            surf = self._text(hud.banner, self.font_md, config.COLOR_TEXT)
            w = surf.get_width() + 48
            panel = self._panel(w, 52, radius=26, alpha=200)
            pr = panel.get_rect(center=(self.width // 2, self.HUD_HEIGHT + 44))
            self.screen.blit(panel, pr.topleft)
            self._blit_text(hud.banner, self.font_md, config.COLOR_WARN, self.width // 2, self.HUD_HEIGHT + 44, center=True)
        if hud.calibrating:
            self._blit_text(f"CALIBRANDO {hud.calibration:0.1f}s", self.font_lg, config.COLOR_ACCENT_2,
                            self.width // 2, self.HUD_HEIGHT + 104, center=True)
            self._blit_text("Abre bien la mano, dedos estirados", self.font_md, config.COLOR_TEXT,
                            self.width // 2, self.HUD_HEIGHT + 142, center=True)
        if hud.fist_progress > 0 and hud.mode != "menu":
            self._fist_ring(self.width // 2, self.height - self.KEY_BAND_HEIGHT - 52, hud.fist_progress,
                            config.COLOR_ACCENT_2)
            self._blit_text("Abriendo menú...", self.font_sm, config.COLOR_TEXT,
                            self.width // 2, self.height - self.KEY_BAND_HEIGHT - 52, center=True)

    def _fist_ring(self, cx: int, cy: int, progress: float, color) -> None:
        radius = 26
        pygame.draw.circle(self.screen, (12, 14, 30), (cx, cy), radius, 4)
        steps = max(1, int(40 * clamp(progress, 0.0, 1.0)))
        points = []
        for i in range(steps + 1):
            a = -math.pi / 2 + math.tau * (i / 40) * clamp(progress, 0.0, 1.0)
            points.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 5)

    def _draw_game_hud(self, game: dict) -> None:
        panel = self._panel(300, 74, radius=16, alpha=190)
        x = self.width // 2 - 150
        self.screen.blit(panel, (x, 8))
        self._blit_text(f"SCORE {game.get('score', 0):>6}", self.font_md, config.COLOR_TEXT, x + 16, 16)
        self._blit_text(f"COMBO x{game.get('combo', 0)}", self.font_sm, config.COLOR_ACCENT, x + 16, 48)
        self._blit_text(f"ACC {game.get('accuracy', 0):5.1f}%", self.font_sm, config.COLOR_TEXT_DIM, x + 160, 48)
        judgement = game.get("last_judgement", "")
        colors = {"perfect": config.COLOR_ACCENT, "good": config.COLOR_OK, "ok": config.COLOR_WARN, "miss": config.COLOR_ACCENT_2}
        if judgement and game.get("judgement_timer", 0) > 0:
            self._blit_text(judgement.upper(), self.font_lg, colors.get(judgement, config.COLOR_TEXT),
                            self.width // 2, self.height - self.KEY_BAND_HEIGHT - 64, center=True)

    # ------------------------------------------------------------------ menú
    def _song_card(self, song: MenuSong, w: int, h: int, selected: bool) -> "pygame.Surface":
        key = ("card", song.title, w, h, selected)
        surf = self._card_cache.get(key)
        if surf is not None:
            return surf
        base = _vertical_gradient(w, h, _lerp_color(song.color, (10, 12, 28), 0.75),
                                  _lerp_color(song.color, (10, 12, 28), 0.35))
        surf = _round_surface(base, 22)
        cover = pygame.Surface((w, int(h * 0.55)), pygame.SRCALPHA)
        bars = song.notes[:10] or config.VISUAL_KEYS
        bw = max(6, (w - 40) // max(1, len(bars)))
        for i, note in enumerate(bars):
            cx = 20 + i * bw + bw // 2
            height = int((0.35 + 0.5 * ((i * 7) % 5) / 4) * h * 0.5)
            color = config.NOTE_COLORS.get(note, song.color)
            pygame.draw.rect(cover, (*color, 210), (cx - bw // 2 + 2, int(h * 0.5) - height, bw - 6, height),
                             border_radius=4)
        surf.blit(cover, (0, 0))
        sheen = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(sheen, (255, 255, 255, 22), (0, 0, w, int(h * 0.3)), border_radius=22)
        surf.blit(sheen, (0, 0))
        edge = song.color if selected else _lerp_color(song.color, (0, 0, 0), 0.3)
        pygame.draw.rect(surf, edge, (0, 0, w, h), 4 if selected else 2, border_radius=22)
        self._card_cache[key] = surf
        return surf

    def draw_menu(self, menu: MenuData) -> None:
        self.screen.blit(self._scrim(30, 210), (0, 0))

        self._blit_text("MANO MUSICAL", self.font_title, config.COLOR_TEXT, 64, 44)
        self._blit_text("ELIGE TU CANCIÓN", self.font_md, config.COLOR_ACCENT, 188, 96)
        pygame.draw.rect(self.screen, config.COLOR_ACCENT_2, (66, 116, 220, 4), border_radius=2)
        if menu.subtitle:
            self._blit_text(menu.subtitle, self.font_sm, config.COLOR_TEXT_DIM, 66, 128)
        self._blit_text("proyecto de Tomas Urieles", self.font_tiny, config.COLOR_TEXT_DIM, 66, 150)

        # estado arriba a la derecha
        ok = menu.camera and menu.tracker
        dot = config.COLOR_OK if ok else config.COLOR_WARN
        pygame.draw.circle(self.screen, dot, (self.width - 210, 52), 7)
        self._blit_text("cámara + tracker" if ok else "usa el teclado",
                        self.font_sm, dot, self.width - 192, 42)

        n = max(1, len(menu.songs))
        card_w, card_h = 210, 270
        gap = 26
        total = n * card_w + (n - 1) * gap
        x0 = (self.width - total) // 2
        base_cy = self.height // 2 - 6
        for i, song in enumerate(menu.songs):
            selected = i == menu.selected
            w = int(card_w * (1.14 if selected else 1.0))
            h = int(card_h * (1.14 if selected else 1.0))
            cx = x0 + card_w // 2 + i * (card_w + gap)
            cy = base_cy - (16 if selected else 0)
            rect = pygame.Rect(0, 0, w, h)
            rect.center = (cx, cy)
            if selected:
                self._glow_rect(rect, song.color, layers=4, radius=24, spread=6)
            self.screen.blit(self._song_card(song, w, h, selected), rect.topleft)
            title_color = config.COLOR_TEXT if selected else config.COLOR_TEXT_DIM
            lines = self._wrap(song.title, self.font_md, w - 28)
            ty = rect.bottom - 84
            for line in lines:
                self._blit_text(line, self.font_md, title_color, rect.centerx, ty, center=True)
                ty += 24
            self._blit_text(f"{song.bpm} BPM · {song.tag}", self.font_tiny,
                            config.COLOR_TEXT_DIM if not selected else config.COLOR_ACCENT,
                            rect.centerx, rect.bottom - 28, center=True)
            if selected:
                self._blit_text("▶", self.font_lg, config.COLOR_TEXT, rect.x + 20, rect.y + 14)

        # selector de velocidad
        y = base_cy + card_h // 2 + 44
        pills = menu.diffs or []
        pw = 130
        total_p = max(1, len(pills)) * pw + (max(1, len(pills)) - 1) * 12
        px = (self.width - total_p) // 2
        for i, name in enumerate(pills):
            active = i == menu.diff_index
            rect = pygame.Rect(px, y, pw, 40)
            color = config.COLOR_ACCENT if active else config.COLOR_TEXT_DIM
            surf = self._panel(pw, 40, radius=20, color=config.COLOR_PANEL, edge=color, alpha=200 if active else 120)
            self.screen.blit(surf, rect.topleft)
            self._blit_text(name, self.font_sm, color, rect.centerx, rect.centery, center=True)
            px += pw + 12

        cy_hint = y + 84
        self._fist_ring(self.width // 2 - 300, cy_hint, menu.fist_progress,
                        config.COLOR_OK if menu.fist_ready else config.COLOR_ACCENT)
        self._blit_text("✊  Mantén el puño ¾ s para jugar", self.font_md, config.COLOR_TEXT, self.width // 2 - 270, cy_hint, center=False)
        self._blit_text("👋  Desliza la mano para cambiar de canción", self.font_sm, config.COLOR_TEXT_DIM,
                        self.width // 2 - 270, cy_hint + 30, center=False)

        self._blit_text("F1 piano libre   ←/→ canción   -/= velocidad   ENTER jugar   ESC salir",
                        self.font_sm, config.COLOR_TEXT_DIM, self.width // 2, self.height - 30, center=True)

        if menu.message and menu.message_timer > 0:
            surf = self._text(menu.message, self.font_md, config.COLOR_TEXT)
            w = surf.get_width() + 48
            panel = self._panel(w, 50, radius=25, alpha=205)
            pr = panel.get_rect(center=(self.width // 2, self.height - 92))
            self.screen.blit(panel, pr.topleft)
            self._blit_text(menu.message, self.font_md, config.COLOR_WARN, self.width // 2, self.height - 92, center=True)

    # ------------------------------------------------------------------ ayuda
    def draw_help(self) -> None:
        self.screen.blit(self._scrim(220, 230), (0, 0))
        panel = self._panel(min(920, self.width - 80), self.height - 120, radius=26)
        pr = panel.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(panel, pr.topleft)
        lines = [
            ("MANO MUSICAL — AYUDA", config.COLOR_ACCENT, self.font_lg),
            ("Baja/dobla un dedo para tocar su nota (air-piano).", config.COLOR_TEXT, self.font_md),
            ("Izq: meñique C4 · anular D4 · medio E4 · índice F4 · pulgar G4", config.COLOR_TEXT_DIM, self.font_sm),
            ("Der: pulgar A4 · índice B4 · medio C5 · anular D5 · meñique E5", config.COLOR_TEXT_DIM, self.font_sm),
            ("", config.COLOR_TEXT, self.font_sm),
            ("MENÚ GESTUAL:  puño ¾ s = abrir menú / jugar   ·   desliza = cambiar canción", config.COLOR_ACCENT_2, self.font_md),
            ("Menú: F1 piano libre   F2 juego   F3 demo   F4 replay   ←/→ canción   -/= velocidad", config.COLOR_TEXT, self.font_sm),
            ("Juego: 1-4 canción   SPACE reinicia   ESC vuelve al menú", config.COLOR_TEXT, self.font_sm),
            ("C calibrar (abre la mano 4s)   X invertir manos   V espejo   B rotar 180°", config.COLOR_WARN, self.font_sm),
            ("R grabar/reproducir   T timbre   M mutear   O octava   , . sensibilidad   F12 captura", config.COLOR_TEXT, self.font_sm),
            ("[ / ] modo de detección (curl/drop)   F10 ayuda   ESC salir", config.COLOR_TEXT, self.font_sm),
        ]
        y = pr.y + 42
        for text, color, font in lines:
            if text:
                self._blit_text(text, font, color, self.width // 2, y, center=True)
            y += 40 if font is not self.font_sm else 30
        self._blit_text("H o F10 para cerrar", self.font_sm, config.COLOR_TEXT_DIM, self.width // 2, pr.bottom - 26, center=True)

    def present(self, dt: float) -> None:
        self.particles.update(dt)
        self.particles.draw(self.screen)
        self.time += dt
        pygame.display.flip()
