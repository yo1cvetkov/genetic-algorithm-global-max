import pygame
import numpy as np

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
POPULATION_SIZE = 500
RECT_SIZE = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
VIRIDIS = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]

# def maxfunc(sample):
#     x, y = sample
#     return (16. * x * (1 - x) * y * (1 - y) * np.sin(15. * np.pi * x) * np.sin(15. * np.pi * y)) ** 2

local_maxima = []

def custom_fitness(sample):

    if not local_maxima:
        return 0
    x, y = sample
    distances = [np.sqrt((x - mx)**2 + (y - my)**2) for mx, my in local_maxima]
    fitness = np.exp(-10 * np.min(distances))  
    return fitness

def rank(sample, fitness):
    fitness /= np.sum(fitness)
    sa = np.argsort(fitness)
    return sample[sa], np.cumsum(fitness[sa])

def select_individuals(acfitness, min_value=0.9):
    return acfitness > min(np.random.uniform(), min_value)

def breed_crossover(parent1, parent2):
    index = int(np.random.uniform(low=1, high=len(parent1) - 1))
    return np.hstack([parent1[:index], parent2[index:]])

def generate_offspring(sample, good_individuals):
    parents = sample[good_individuals]
    nr_of_children = np.sum(~good_individuals)
    Np, Ni = parents.shape
    children = np.zeros((nr_of_children, Ni))
    index1 = np.array(np.random.uniform(size=nr_of_children, high=Np - 1).round(), int)
    index2 = np.array(np.random.uniform(size=nr_of_children, high=Np - 1).round(), int)
    for nr, (i, j) in enumerate(zip(index1, index2)):
        children[nr] = breed_crossover(parents[i], parents[j])
    return children

def evolve(sample):
    fitness = np.array([custom_fitness(individual) for individual in sample])
    sample, fitness_ = rank(sample, fitness.copy())
    good = select_individuals(fitness_)
    bad = np.where(~good)[0]
    children = generate_offspring(sample, good)
    sample[bad[:len(children)]] = children
    return sample

def draw_rectangles(screen, sample):
    screen.fill(BLACK)
    max_fitness = custom_fitness((0.5, 0.5)) if local_maxima else 1 
    for (x, y) in sample:
        fitness_value = custom_fitness((x, y))
        if max_fitness > 0:
            color_index = int((fitness_value / max_fitness) * (len(VIRIDIS) - 1))
            color_index = max(0, min(color_index, len(VIRIDIS) - 1))
        else:
            color_index = 0
        color = VIRIDIS[color_index]
        pygame.draw.rect(screen, color, (x * SCREEN_WIDTH, y * SCREEN_HEIGHT, RECT_SIZE, RECT_SIZE))
    pygame.display.flip()

def main():
    global local_maxima
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Genetic algorithm - finding global maximum')

    sample = np.random.uniform(size=(POPULATION_SIZE, 2), low=0, high=1)
    draw_rectangles(screen, sample)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x,y = pygame.mouse.get_pos()
                local_maxima.append((x / SCREEN_WIDTH, y / SCREEN_HEIGHT))
                draw_rectangles(screen, sample)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and local_maxima:
                    sample = evolve(sample)
                    draw_rectangles(screen, sample)
                if event.key == pygame.K_i:
                    sample = np.random.uniform(size=(POPULATION_SIZE, 2), low=0, high=1)
                    draw_rectangles(screen, sample)

    pygame.quit()

if __name__ == "__main__":
    main()
