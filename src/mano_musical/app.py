"""Orquestador principal: une captura, inferencia, audio, juego y render.

El bucle principal corre en el hilo de pygame. La cámara y MediaPipe corren en
hilos aparte y publican su resultado en ``SharedState``. Los eventos de nota se
drenan en el hilo principal para disparar audio, partículas y puntaje.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Optional

from . import config, music
from .audio_engine import AudioEngine
from .capture import CameraThread, InferenceThread
from .game import RhythmGame, SongPlayer
from .gesture import FingerPressDetector, MenuGestureController
from .hand_tracker import HandTracker
from .recorder import Performance, PerformancePlayer, PerformanceRecorder, save_performance
from .render import HudData, MenuData, MenuSong, Renderer
from .state import SharedState
from .utils import setup_logging

log = logging.getLogger("mano_musical.app")

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None  # type: ignore[assignment]

TIMBRES = ["piano", "organ", "chiptune", "synth"]


@dataclass
class UiState:
    mode: str = "free"
    show_help: bool = False
    banner: str = ""
    banner_timer: float = 0.0
    game: Optional[dict] = None
    judgement_timer: float = 0.0
    last_judgement: str = ""


class ManoMusicalApp:
    """Aplicación air-piano completa."""

    def __init__(self, args: argparse.Namespace) -> None:
        if pygame is None:  # pragma: no cover
            raise RuntimeError("pygame es requerido para ejecutar la app")
        setup_logging(args.debug)
        self.args = args
        pygame.mixer.pre_init(
            frequency=config.AUDIO_FREQUENCY,
            size=config.AUDIO_SIZE,
            channels=config.AUDIO_CHANNELS,
            buffer=config.AUDIO_BUFFER,
        )
        pygame.init()
        pygame.display.set_caption(config.WINDOW_TITLE)

        self.renderer = Renderer(args.width, args.height)
        self.clock = pygame.time.Clock()
        self.audio = AudioEngine(timbre=args.timbre, enabled=not args.mute)
        self.state = SharedState()
        self.state.running.set()
        self.tracker = HandTracker(swap_handedness=args.swap_hands)
        self.state.tracker_available = self.tracker.available
        self.detector = FingerPressDetector(mode=args.detection, swap_hands=False)
        self.menu_ctrl = MenuGestureController()
        self._fist_hands: set[str] = set()
        self._fist_progress = 0.0
        self.rotate180 = bool(args.rotate180)

        self.camera = None if args.no_camera else CameraThread(
            self.state, index=args.camera, mirror=not args.no_mirror, rotate180=self.rotate180
        )
        self.inference = (
            InferenceThread(self.state, self.tracker, self.detector) if self.tracker.available else None
        )

        self.song_index = 0
        self.diff_index = 1
        self.game = RhythmGame(self._current_song())
        self.autoplay = SongPlayer(self._current_song(), loop=True)
        self._menu_songs = self._build_menu_songs()
        self.recorder = PerformanceRecorder()
        self.replayer: Optional[PerformancePlayer] = None
        self.last_performance: Optional[Performance] = None
        self._game_reported = False

        self.voices: dict[str, tuple[str, float]] = {}
        self.active_notes: dict[str, int] = {}
        self.keyboard_held: set[str] = set()
        self.octave = 0
        self.timbre_index = TIMBRES.index(args.timbre) if args.timbre in TIMBRES else 0
        self.ui = UiState(mode=args.mode)
        self.running = True
        self._apply_mode(args.mode)

    # -------------------------------------------------------------- lifecycle
    def run(self) -> int:
        log.info("Iniciando Mano Musical (mode=%s, cámara=%s)", self.ui.mode, not self.args.no_camera)
        try:
            if self.camera:
                self.camera.start()
            if self.inference:
                self.inference.start()
            self._show_banner("¡Bienvenido! Puño = menú · desliza = canción · H = ayuda", 4.0)
            self._loop()
        finally:
            self.state.running.clear()
            if self.camera:
                self.camera.join(timeout=1.0)
            if self.inference:
                self.inference.join(timeout=1.0)
            self.audio.shutdown()
            self.tracker.close()
            pygame.quit()
        return 0

    def _loop(self) -> None:
        prev = time.perf_counter()
        while self.running:
            dt = self.clock.tick(config.TARGET_FPS) / 1000.0
            now = time.perf_counter()
            self._handle_events(now)
            self._update_gestures(now)
            self._drain_events(now)
            self._update_autoplay(now)
            self._update_replay(now)
            self._update_game(now)
            if self.ui.banner_timer > 0:
                self.ui.banner_timer = max(0.0, self.ui.banner_timer - dt)
            if self.ui.judgement_timer > 0:
                self.ui.judgement_timer = max(0.0, self.ui.judgement_timer - dt)
            self._render(now, dt)
            prev = now

    # ------------------------------------------------------------------ events
    def _handle_events(self, now: float) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._on_keydown(event, now)
            elif event.type == pygame.KEYUP:
                self._on_keyup(event)
            elif event.type == pygame.VIDEORESIZE:
                size = (event.w, event.h)
                self.renderer = Renderer(max(640, size[0]), max(480, size[1]))

    def _on_keydown(self, event, now: float) -> None:
        key = event.key
        if self.ui.mode == "menu" and key in (
            pygame.K_LEFT, pygame.K_RIGHT, pygame.K_a, pygame.K_d,
            pygame.K_UP, pygame.K_DOWN, pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE,
        ):
            if key in (pygame.K_RIGHT, pygame.K_d):
                self._select_song((self.song_index + 1) % len(music.SONGS), now)
            elif key in (pygame.K_LEFT, pygame.K_a):
                self._select_song((self.song_index - 1) % len(music.SONGS), now)
            elif key == pygame.K_UP:
                self._cycle_diff(1, now)
            elif key == pygame.K_DOWN:
                self._cycle_diff(-1, now)
            else:
                self._start_game(now)
            return
        if key in (pygame.K_ESCAPE,):
            if self.ui.mode == "menu":
                self.running = False
            else:
                self._apply_mode("menu", now)
        elif key in (pygame.K_F10, pygame.K_SLASH):
            self.ui.show_help = not self.ui.show_help
        elif key in (pygame.K_F1,):
            self._apply_mode("free", now)
        elif key in (pygame.K_F2,):
            self._apply_mode("game", now)
        elif key in (pygame.K_F3,):
            self._apply_mode("demo", now)
        elif key in (pygame.K_F4,):
            self._apply_mode("replay", now)
        elif key == pygame.K_r:
            self._toggle_record(now)
        elif key == pygame.K_t:
            self.timbre_index = (self.timbre_index + 1) % len(TIMBRES)
            self.audio.set_timbre(TIMBRES[self.timbre_index])
            self._show_banner(f"Timbre: {TIMBRES[self.timbre_index]}")
        elif key == pygame.K_m:
            self._muted = not getattr(self, "_muted", False)
            self.audio.set_master_volume(0.0 if self._muted else config.MASTER_VOLUME)
            self._show_banner("Silencio" if self._muted else "Audio activo")
        elif key == pygame.K_o:
            self.octave = 0 if self.octave >= 1 else self.octave + 1
            self._show_banner(f"Octava {'+' if self.octave else '0'}")
        elif key == pygame.K_c:
            self.detector.begin_calibration(now)
            self._show_banner(f"Calibrando {config.CALIBRATION_SECONDS:.0f}s: abre bien la mano", 4.0)
        elif key == pygame.K_x:
            self.tracker.swap_handedness = not self.tracker.swap_handedness
            self.detector.reset()
            self._show_banner(f"Manos {'invertidas' if self.tracker.swap_handedness else 'normales'}")
        elif key == pygame.K_v:
            if self.camera:
                self.camera.mirror = not self.camera.mirror
                self.detector.reset()
            self._show_banner(f"Espejo {'ON' if (self.camera and self.camera.mirror) else 'OFF'}")
        elif key == pygame.K_b:
            self.rotate180 = not self.rotate180
            if self.camera:
                self.camera.rotate180 = self.rotate180
                self.detector.reset()
            self._show_banner(f"Rotación 180° {'ON' if self.rotate180 else 'OFF'}")
        elif key == pygame.K_COMMA:
            self.detector.nudge_sensitivity(-config.SENSITIVITY_STEP)
            self._show_banner(f"Sensibilidad {self.detector.press_enter:.0f}° (más fácil bajar)")
        elif key == pygame.K_PERIOD:
            self.detector.nudge_sensitivity(config.SENSITIVITY_STEP)
            self._show_banner(f"Sensibilidad {self.detector.press_enter:.0f}° (más difícil)")
        elif key == pygame.K_LEFTBRACKET:
            self.detector.mode = "curl" if self.detector.mode != "curl" else "drop"
            self.detector.reset()
            self._show_banner(f"Detección: {self.detector.mode}")
        elif key == pygame.K_RIGHTBRACKET:
            self.detector.mode = "drop" if self.detector.mode != "drop" else "curl"
            self.detector.reset()
            self._show_banner(f"Detección: {self.detector.mode}")
        elif key == pygame.K_SPACE and self.ui.mode == "game":
            self._apply_mode("game", time.perf_counter())
        elif pygame.K_1 <= key <= pygame.K_1 + len(music.SONGS) - 1:
            self._select_song(key - pygame.K_1, now)
        elif key == pygame.K_MINUS:
            self._cycle_diff(-1, now)
        elif key == pygame.K_EQUALS:
            self._cycle_diff(1, now)
        elif key == pygame.K_F12:
            self._save_screenshot()

        name = pygame.key.name(key)
        note = config.KEYBOARD_MAP.get(name)
        if note and self.ui.mode != "menu" and name not in self.keyboard_held:
            self.keyboard_held.add(name)
            self._trigger_on(note, 1.0, now, voice=f"kb:{name}")

    def _on_keyup(self, event) -> None:
        name = pygame.key.name(event.key)
        if name in self.keyboard_held:
            self.keyboard_held.discard(name)
            note = config.KEYBOARD_MAP.get(name)
            if note:
                self._trigger_off(note, voice=f"kb:{name}")

    # --------------------------------------------------------------- triggers
    def _trigger_on(self, note: str, velocity: float, now: float, voice: str) -> None:
        sounded = music.transpose_note(note, self.octave * 12)
        self.audio.note_on(voice, sounded, velocity)
        self.voices[voice] = (note, velocity)
        self.active_notes[note] = self.active_notes.get(note, 0) + 1
        if self.recorder.recording:
            self.recorder.record(note, velocity, "on", now)
        if self.ui.mode == "game":
            result = self.game.register_input(note, now)
            if result:
                self.ui.last_judgement = result.judgement
                self.ui.judgement_timer = 0.8
        self._burst_for(note, velocity)

    def _trigger_off(self, note: str, voice: str) -> None:
        self.audio.note_off(voice)
        self.voices.pop(voice, None)
        if note in self.active_notes:
            self.active_notes[note] -= 1
            if self.active_notes[note] <= 0:
                self.active_notes.pop(note, None)

    def _burst_for(self, note: str, velocity: float) -> None:
        reading = self.detector.last_readings.get(note)
        if reading and "tip" in reading:
            pos = self.renderer.to_screen(*reading["tip"])
        else:
            rect = self.renderer.key_rect(note)
            pos = (rect.centerx, rect.top)
        self.renderer.burst_note(note, pos, velocity)

    def _drain_events(self, now: float) -> None:
        pressed_counts = self._hand_pressed_counts()
        for trigger in self.state.drain_events():
            if self.ui.mode == "menu":
                continue
            if trigger.kind == "on" and (
                trigger.hand in self._fist_hands or pressed_counts.get(trigger.hand, 0) >= 4
            ):
                continue
            voice = f"{trigger.hand}:{trigger.finger}"
            if trigger.kind == "on":
                self._trigger_on(trigger.note, trigger.velocity, now, voice)
            else:
                self._trigger_off(trigger.note, voice)

    def _hand_pressed_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for reading in self.detector.last_readings.values():
            if reading.get("pressed"):
                hand = reading.get("hand", "?")
                counts[hand] = counts.get(hand, 0) + 1
        return counts

    # -------------------------------------------------------------- gestos menú
    def _update_gestures(self, now: float) -> None:
        hand_frame = self.state.get_hand_frame()
        commands = self.menu_ctrl.update(hand_frame, now)
        self._fist_hands = {key for key, is_fist in self.menu_ctrl.fist_state.items() if is_fist}
        self._fist_progress = max(self.menu_ctrl.progress.values(), default=0.0)
        for command in commands:
            if command.kind == "fist":
                self._on_fist(now)
            elif self.ui.mode == "menu" and command.kind == "next":
                self._select_song((self.song_index + 1) % len(music.SONGS), now)
            elif self.ui.mode == "menu" and command.kind == "prev":
                self._select_song((self.song_index - 1) % len(music.SONGS), now)

    def _on_fist(self, now: float) -> None:
        if self.ui.mode == "menu":
            self._start_game(now)
        else:
            self._apply_mode("menu", now)

    def _start_game(self, now: float) -> None:
        self._apply_mode("game", now)

    def _build_menu_songs(self) -> list[MenuSong]:
        tags = ["media", "fácil", "muy fácil", "intermedia"]
        colors = [config.COLOR_ACCENT, config.COLOR_WARN, config.COLOR_ACCENT_2, config.COLOR_PURPLE]
        songs: list[MenuSong] = []
        for i, factory in enumerate(music.SONGS):
            song = factory()
            songs.append(
                MenuSong(
                    title=song.name,
                    bpm=song.bpm,
                    tag=tags[i] if i < len(tags) else "media",
                    notes=song.used_notes(),
                    color=colors[i % len(colors)],
                )
            )
        return songs

    # ------------------------------------------------------------------ modes
    def _current_song(self) -> music.Song:
        base = music.SONGS[self.song_index]()
        return music.scale_tempo(base, config.DIFFICULTIES[self.diff_index][1])

    def _rebuild_songs(self) -> None:
        song = self._current_song()
        self.game = RhythmGame(song)
        self.autoplay = SongPlayer(song, loop=True)

    def _select_song(self, index: int, now: float) -> None:
        self.song_index = index
        self._restart_songs(now)

    def _cycle_diff(self, step: int, now: float) -> None:
        self.diff_index = (self.diff_index + step) % len(config.DIFFICULTIES)
        self._restart_songs(now)

    def _restart_songs(self, now: float) -> None:
        self._rebuild_songs()
        if self.ui.mode in ("game", "demo"):
            self._apply_mode(self.ui.mode, now)
        self._show_banner(
            f"{self.game.song.name} · velocidad {config.DIFFICULTIES[self.diff_index][0]} (F2 para jugar)"
        )

    def _apply_mode(self, mode: str, now: Optional[float] = None) -> None:
        now = now or time.perf_counter()
        self.ui.mode = mode
        self.autoplay.stop()
        if self.replayer:
            self.replayer.stop()
        self._release_all()
        if mode == "game":
            self._game_reported = False
            self.game.reset()
            self.game.start(now)
            self._show_banner("Modo JUEGO: baja el dedo (o tecla) cuando el ladrillo toque la línea. SPACE reinicia")
        elif mode == "demo":
            self.game.reset()
            self.autoplay.start(now)
            self._show_banner(f"Demo: {self.game.song.name} automática")
        elif mode == "replay":
            if self.last_performance:
                self.replayer = PerformancePlayer(self.last_performance, loop=True)
                self.replayer.start(now)
                self._show_banner("Reproduciendo tu grabación")
            else:
                self._show_banner("Nada grabado aún (R para grabar)")
        elif mode == "menu":
            self.menu_ctrl.reset()
            self._show_banner("Menú: desliza para elegir · puño para jugar")
        else:
            self._show_banner("Modo libre: improvisa")

    def _release_all(self) -> None:
        for voice in list(self.voices):
            note = self.voices[voice][0]
            self.audio.note_off(voice)
            if note in self.active_notes:
                self.active_notes[note] -= 1
                if self.active_notes[note] <= 0:
                    self.active_notes.pop(note, None)
        self.voices.clear()
        self.keyboard_held.clear()

    def _toggle_record(self, now: float) -> None:
        if self.recorder.recording:
            perf = self.recorder.stop("airpiano")
            path = save_performance(perf)
            self.last_performance = perf
            self._show_banner(f"Grabación guardada ({len(perf.events)} notas)")
            log.info("Grabación -> %s", path)
        else:
            self.recorder.start(now)
            self._show_banner("Grabando... pulsa R para detener")

    # ----------------------------------------------------------------- update
    def _update_autoplay(self, now: float) -> None:
        if not self.autoplay.playing:
            return
        for note, duration in self.autoplay.update(now):
            sounded = music.transpose_note(note, self.octave * 12)
            self.audio.note_on(f"auto:{note}:{now:.4f}", sounded, 0.9)
            self._burst_for(note, 0.8)

    def _update_replay(self, now: float) -> None:
        if not self.replayer or not self.replayer.playing:
            return
        for event in self.replayer.update(now):
            voice = f"replay:{event.note}:{event.time:.3f}:{event.kind}"
            if event.kind == "on":
                sounded = music.transpose_note(event.note, self.octave * 12)
                self.audio.note_on(voice, sounded, event.velocity)
                self._burst_for(event.note, event.velocity)
            else:
                self.audio.note_off(voice)

    def _update_game(self, now: float) -> None:
        if self.ui.mode != "game":
            self.ui.game = None
            return
        for result in self.game.update(now):
            if result.judgement == "miss":
                self.ui.last_judgement = "miss"
                self.ui.judgement_timer = 0.6
        if self.game.finished and not self._game_reported:
            self._game_reported = True
            st = self.game.state
            self._show_banner(
                f"¡Fin! {self.game.song.name} · Score {st.score} · Precisión {self.game.accuracy():.0f}% "
                f"· Combo máx {st.max_combo} · Perfect {st.counts['perfect']} Good {st.counts['good']} "
                f"OK {st.counts['ok']} Miss {st.counts['miss']} (SPACE reinicia)",
                12.0,
            )
        self.ui.game = {
            "score": self.game.state.score,
            "combo": self.game.state.combo,
            "accuracy": self.game.accuracy(),
            "last_judgement": self.ui.last_judgement,
            "judgement_timer": self.ui.judgement_timer,
            "song": self.game.song.name,
            "diff": config.DIFFICULTIES[self.diff_index][0],
        }

    # ------------------------------------------------------------------ render
    def _render(self, now: float, dt: float) -> None:
        frame, _, _ = self.state.get_frame()
        hand_frame = self.state.get_hand_frame()
        if self.ui.mode == "menu":
            self._render_menu(frame, hand_frame, dt)
            return
        self.renderer.draw_background()
        self.renderer.draw_camera(frame)
        self.renderer.draw_hands(hand_frame, self.detector.last_readings)
        if self.ui.mode == "game":
            horizon = config.GAME_FALL_HORIZON
            self.renderer.draw_falling_notes(self.game.upcoming(now, horizon), horizon)
        self.renderer.draw_keys(list(self.active_notes.keys()))
        hud = HudData(
            mode=self.ui.mode,
            fps=self.clock.get_fps(),
            infer_ms=self.state.stats.infer_ms,
            hands=self.state.stats.hands,
            active_notes=list(self.active_notes.keys()),
            camera=self.state.camera_available,
            tracker=self.state.tracker_available,
            timbre=TIMBRES[self.timbre_index],
            octave=self.octave,
            recording=self.recorder.recording,
            banner=self.ui.banner,
            banner_timer=self.ui.banner_timer,
            countdown=self.game.countdown(now) if self.ui.mode == "game" and self.game.start_time else 0.0,
            game=self.ui.game,
            detection=self.detector.mode,
            mirror=bool(self.camera and self.camera.mirror),
            rotate=self.rotate180,
            swapped=self.tracker.swap_handedness,
            calibrating=self.detector.calibrating,
            calibration=self.detector.calibration_remaining(now),
            sensitivity=self.detector.press_enter,
            fist_progress=self._fist_progress,
        )
        self.renderer.draw_hud(hud)
        if self.ui.show_help:
            self.renderer.draw_help()
        self.renderer.present(dt)

    def _render_menu(self, frame, hand_frame, dt: float) -> None:
        self.renderer.draw_background()
        if frame is not None:
            full = pygame.Rect(0, 0, self.renderer.width, self.renderer.height)
            self.renderer.draw_camera(frame, rect=full, radius=0, dim=120, fill=True)
            self.renderer.draw_hands(hand_frame, self.detector.last_readings)
        menu = MenuData(
            songs=self._menu_songs,
            selected=self.song_index,
            diffs=[name for name, _ in config.DIFFICULTIES],
            diff_index=self.diff_index,
            hands=self.state.stats.hands,
            camera=self.state.camera_available,
            tracker=self.state.tracker_available,
            fist_progress=self._fist_progress,
            fist_ready=bool(self._fist_hands),
            message=self.ui.banner,
            message_timer=self.ui.banner_timer,
            subtitle="Air-Piano con visión por computador · baja los dedos y suenan las notas",
        )
        self.renderer.draw_menu(menu)
        if self.ui.show_help:
            self.renderer.draw_help()
        self.renderer.present(dt)

    # ---------------------------------------------------------------- helpers
    def _save_screenshot(self) -> None:
        out = config.PROJECT_ROOT / "docs" / "img"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"captura_{time.strftime('%Y%m%d_%H%M%S')}.png"
        pygame.image.save(self.renderer.screen, str(path))
        self._show_banner(f"Captura: {path.relative_to(config.PROJECT_ROOT)}")
        log.info("Captura -> %s", path)

    def _show_banner(self, text: str, seconds: float = 2.5) -> None:
        self.ui.banner = text
        self.ui.banner_timer = seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mano-musical",
        description="Air-piano con visión por computador que reconoce tus dedos y toca Jingle Bells.",
    )
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="índice de cámara")
    parser.add_argument("--width", type=int, default=config.WINDOW_WIDTH)
    parser.add_argument("--height", type=int, default=config.WINDOW_HEIGHT)
    parser.add_argument("--mode", choices=["menu", "free", "game", "demo", "replay"], default=config.DEFAULT_MODE)
    parser.add_argument("--timbre", choices=TIMBRES, default="piano")
    parser.add_argument("--detection", choices=["curl", "drop"], default="curl")
    parser.add_argument("--no-camera", action="store_true", help="modo teclado sin cámara")
    parser.add_argument("--no-mirror", action="store_true", help="no espejar la imagen")
    parser.add_argument("--rotate180", action="store_true", help="rotar la imagen 180°")
    parser.add_argument("--swap-hands", dest="swap_hands", action="store_true", help="invertir izquierda/derecha")
    parser.add_argument("--no-swap-hands", dest="swap_hands", action="store_false", help="no invertir izquierda/derecha")
    parser.set_defaults(swap_hands=config.DEFAULT_SWAP_HANDS)
    parser.add_argument("--mute", action="store_true", help="arrancar en silencio")
    parser.add_argument("--debug", action="store_true")
    return parser
