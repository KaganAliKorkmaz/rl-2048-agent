import pygame
from Game_2048_BE import Game2048Env  # sınıfı buraya koyduğunu varsayıyorum

TILE_SIZE = 100
GRID_SIZE = 4
MARGIN = 10
WINDOW_SIZE = GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1) * MARGIN

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("2048")
font = pygame.font.SysFont("Arial", 32, bold=True)

def draw_board(env):
    screen.fill((187, 173, 160))  # arka plan
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            val = env.board[i][j]
            x = MARGIN + j * (TILE_SIZE + MARGIN)
            y = MARGIN + i * (TILE_SIZE + MARGIN)
            # basit tek renk – istersen sonra value'ya göre renk paleti eklersin
            pygame.draw.rect(screen, (205, 193, 180), (x, y, TILE_SIZE, TILE_SIZE))
            if val != 0:
                text = font.render(str(val), True, (0, 0, 0))
                text_rect = text.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
                screen.blit(text, text_rect)

def main():
    env = Game2048Env()
    env.reset()
    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)  # FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN and not env.done:
                if event.key == pygame.K_UP:
                    env.step(0)
                elif event.key == pygame.K_DOWN:
                    env.step(1)
                elif event.key == pygame.K_LEFT:
                    env.step(2)
                elif event.key == pygame.K_RIGHT:
                    env.step(3)

        draw_board(env)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
