#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

char mensaje[243]; // {(000,000,000),(000,000,000)...}
int filas, columnas;
const char *imagenTXT = "imagen_rgb.txt";
int numPixeles;
const char *coordenasTXT = "coordenadas.txt";

typedef struct
{
    int r, g, b, x, y;
} RGB;

int aleatorioHastaMax(int min, int max)
{
    return rand() % (max - min + 1) + min;
}

void obtenerDimensionesXY(const char *coordenasTXT, int *numPixeles)
{
    FILE *archivo = fopen(coordenasTXT, "r");
    if (archivo == NULL)
    {
        perror("Error al abrir el archivo");
        exit(EXIT_FAILURE);
    }

    *numPixeles = 0;
    char c;

    while ((c = fgetc(archivo)) != EOF)
    {
        if (c == '(')
        {
            (*numPixeles)++;
        }
    }

    fclose(archivo);
}

void obtenerDimensionesRGB(const char *nombre_archivo, int *filas, int *columnas)
{
    FILE *archivo = fopen(nombre_archivo, "r");
    if (archivo == NULL)
    {
        perror("Error al abrir el archivo");
        exit(EXIT_FAILURE);
    }

    *filas = 0;
    *columnas = 0;
    char c;
    int comienzoLinea = 1; // indicador de nueva linea
    int tempCols = 0;
    int contandorColumnas = 1; // solo contamos columnas en la primera línea (después cambia a 0)

    // Leer caracter por caracter para contar
    while ((c = fgetc(archivo)) != EOF)
    {
        if (comienzoLinea && c != ' ' && c != '\t')
        {
            (*filas)++;
            comienzoLinea = 0;
        }

        if (c == '(' && contandorColumnas)
        {
            tempCols++;
        }

        if (c == '\n')
        {
            comienzoLinea = 1;
            contandorColumnas = 0; // Ya contamos las columnas de la primera línea
        }
    }
    *columnas = tempCols;

    fclose(archivo);
}

RGB **leerDatosRGB(const char *imagenTXT, int filas, int columnas)
{
    FILE *archivo = fopen(imagenTXT, "r");
    if (archivo == NULL)
    {
        perror("Error al abrir el archivo");
        exit(EXIT_FAILURE);
    }

    RGB **imagenRGB = (RGB **)malloc(filas * sizeof(RGB *)); // matriz de RGB de imagen_rgb.txt
    if (imagenRGB == NULL)
    {
        perror("Error de asignación de memoria para filas");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < filas; i++)
    { // Asignar memoria para las columnas de cada fila
        imagenRGB[i] = (RGB *)malloc(columnas * sizeof(RGB));
        if (imagenRGB[i] == NULL) // liberar memoria si imagenRGB es nulo
        {
            perror("Error de asignación de memoria para columnas");
            for (int k = 0; k < i; k++)
                free(imagenRGB[k]);
            free(imagenRGB);
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < filas; i++)
    { // Leer los valores usando fscanf con el formato de tupla (R,G,B)
        for (int j = 0; j < columnas; j++)
        {
            if (fscanf(archivo, " (%d,%d,%d)", &imagenRGB[i][j].r, &imagenRGB[i][j].g, &imagenRGB[i][j].b) != 3)
            {
                fprintf(stderr, "Error al leer datos en la fila %d, columna %d\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(archivo);
    return imagenRGB;
}

RGB *leerDatosXY(const char *coordenasTXT, int numPixeles) // Kenia
{
    FILE *archivo = fopen(coordenasTXT, "r");
    if (archivo == NULL)
    {
        perror("Error al abrir el archivo");
        exit(EXIT_FAILURE);
    }

    RGB *arrASCII = (RGB *)malloc(numPixeles * sizeof(RGB)); // matriz de RGB de imagen_rgb.txt
    if (arrASCII == NULL)
    {
        perror("Error de asignación de memoria para el arreglo");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numPixeles; i++)
    { // Leer los valores usando fscanf con el formato de tupla (x,y)
        if (fscanf(archivo, "(%d,%d) ", &arrASCII[i].x, &arrASCII[i].y) != 2)
        {
            fprintf(stderr, "Error al leer datos en la fila %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    fclose(archivo);
    return arrASCII;
}

void limpiarBuffer()
{
    int c;
    while ((c = getchar()) != '\n' && c != EOF)
        ;
}

enum opciones
{
    Ingresar = 1,
    Leer,
    Salir
};

void blurAlredPonderado(RGB **img, int filas, int columnas, int x, int y, float pesoCentral) // Pauli
{
    RGB central = img[y][x];

    int inicioX = (x > 0) ? x - 1 : x;
    int finX = (x < columnas - 1) ? x + 1 : x;
    int inicioY = (y > 0) ? y - 1 : y;
    int finY = (y < filas - 1) ? y + 1 : y;

    float sumaR = central.r * pesoCentral;
    float sumaG = central.g * pesoCentral;
    float sumaB = central.b * pesoCentral;
    float count = pesoCentral;

    // sumar vecinos
    for (int i = inicioY; i <= finY; i++)
    {
        for (int j = inicioX; j <= finX; j++)
        {
            if (i == y && j == x)
                continue;
            sumaR += img[i][j].r;
            sumaG += img[i][j].g;
            sumaB += img[i][j].b;
            count += 1.0f;
        }
    }

    int promR = sumaR / count;
    int promG = sumaG / count;
    int promB = sumaB / count;

    // asigna el promedio a los vecinos
    for (int i = inicioY; i <= finY; i++)
    {
        for (int j = inicioX; j <= finX; j++)
        {
            if (i == y && j == x)
                continue;
            img[i][j].r = promR;
            img[i][j].g = promG;
            img[i][j].b = promB;
        }
    }
}

void ingresarMensaje();

void leerMensaje();

int main()
{
    int opcion;
    char repetir;

    printf("\n\tInstrucciones: \n"
           "\t\tTener todo los archivos en la misma carpeta\n"
           "\t\tNombrar la imagen en donde se pondrá el mensaje como \"imagen_original.png\"\n"
           "\t\tRecuerde guardar la imagen con el mensaje como \"imagen_mensaje.png\": \n\n");

    printf("\tFormas de uso (ejecute los programas en la terminal): \n"
           "\t    [A] Ingresar un mensaje en una imagen y leerlo inmediatamente\n"
           "\t\t1. imagen_original.py\n"
           "\t\t2. mensajeCifrado.c (opción 1 y 2)\n"
           "\t\t3. recronstruir_desdetxt.py \n\n");
    printf("\t    [B] Leer una imagen con su archivo de coordenadas\n"
           "\t\t1. imagen_mensaje.py\n"
           "\t\t2. mensajeCifrado.c (opción 2)\n");

    do
    {
        printf("\n\t--------MENU--------\n");
        printf("\t1. Ingresar un mensaje\n");
        printf("\t2. Leer mensaje\n");
        printf("\t3. Salir\n");
        printf("\tSeleccione una opción: ");
        scanf("%d", &opcion);
        limpiarBuffer();

        switch (opcion)
        {
        case Ingresar:
            ingresarMensaje();
            break;
        case Leer:
            leerMensaje();
            break;
        case Salir:
            printf("\n\t⇉ Saliendo del programa...\n");
            return 0;
        default:
            printf("\n\tOpcion invalida. Intente de nuevo.\n");
        }

        printf("\n\t↳ ¿Desea volver al menu? (s/n): ");
        scanf("%c", &repetir);
        limpiarBuffer();
    } while (repetir == 's' || repetir == 'S');

    printf("\n\t⇉ Programa finalizado.\n");
    return 0;
}

void ingresarMensaje()
{
    printf("\n\t--------INGRESO--------\n");
    printf("\tEscribe tu mensaje (hasta 243 caracteres, sin acentos): ");

    fgets(mensaje, sizeof(mensaje), stdin);
    mensaje[strcspn(mensaje, "\n")] = '\0'; // sustituir salto de línea con valor nulo
    int longitudMensaje = strlen(mensaje);  // tamaño del mensaje

    for (int i = 0; i < longitudMensaje; i++)
    {
        printf("\t%c = %03d\n", mensaje[i], mensaje[i]); // correspondencia de mensaje con valor ASCII
    }

    int numeroPixeles = (longitudMensaje + 2) / 3; // monse
    RGB *arrMensajeRGB = (RGB *)malloc(numeroPixeles * sizeof(RGB));
    if (arrMensajeRGB == NULL)
    {
        printf("Error de asignación de memoria");
        exit(EXIT_FAILURE);
    }

    int j = 0;
    for (int i = 0; i < longitudMensaje; i += 3)
    {
        arrMensajeRGB[j].r = mensaje[i];
        arrMensajeRGB[j].g = (i + 1 < longitudMensaje) ? mensaje[i + 1] : '\0';
        arrMensajeRGB[j].b = (i + 2 < longitudMensaje) ? mensaje[i + 2] : '\0';
        j++;
    }

    printf("\n\tMensaje traducido a ASCII como tupla RGB:\n\t");
    for (int i = 0; i < numeroPixeles; i++)
    {
        printf("(%03d, %03d, %03d) ", arrMensajeRGB[i].r, arrMensajeRGB[i].g, arrMensajeRGB[i].b); // 000 001
    }

    obtenerDimensionesRGB(imagenTXT, &filas, &columnas);

    //--------------------------------------------------------------------------------
    srand(time(NULL));

    int *usados = calloc(filas * columnas, sizeof(int));
    if (!usados)
    {
        perror("Error asignando memoria");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numeroPixeles; i++)
    {
        int x, y, index;
        do
        {
            x = aleatorioHastaMax(0, columnas - 1);
            y = aleatorioHastaMax(0, filas - 1);
            index = y * columnas + x;
        } while (usados[index] == 1);
        usados[index] = 1;
        arrMensajeRGB[i].x = x;
        arrMensajeRGB[i].y = y;
    }

    // -------------------------------------------------------------------------

    RGB **matrizImagenRGB = leerDatosRGB(imagenTXT, filas, columnas);
    printf("\n");
    for (int i = 0; i < numeroPixeles; i++) // Kenia
    {
        int x = arrMensajeRGB[i].x, y = arrMensajeRGB[i].y;
        matrizImagenRGB[y][x].r = arrMensajeRGB[i].r;
        matrizImagenRGB[y][x].g = arrMensajeRGB[i].g;
        matrizImagenRGB[y][x].b = arrMensajeRGB[i].b;
        printf("\n\tPixel (%d,%d) actualizado como: RGB(%d,%d,%d)",
               x, y,
               matrizImagenRGB[y][x].r,
               matrizImagenRGB[y][x].g,
               matrizImagenRGB[y][x].b);
    }
    printf("\n");
    float pesoCentral = 3.0; // se aplica el blur, con el central teniendo .2 veces más peso que un vecino normal MAYOR PARA DEMOSTRACIÓN
    for (int i = 0; i < numeroPixeles; i++)
    {
        blurAlredPonderado(matrizImagenRGB, filas, columnas, arrMensajeRGB[i].x, arrMensajeRGB[i].y, pesoCentral); // Pauli
    }
    //----------------------------------------------------------------------------------
    fclose(fopen("imagen_rgb.txt", "w")); // abre y cierra el documento, eliminando el contenido Alondra

    FILE *archivoImagen = fopen("imagen_rgb.txt", "w");
    if (archivoImagen == NULL)
    {
        perror("Error al abrir el archivo");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < filas; i++)
    {
        for (int j = 0; j < columnas; j++)
        {
            fprintf(archivoImagen, "(%d,%d,%d) ",
                    matrizImagenRGB[i][j].r,
                    matrizImagenRGB[i][j].g,
                    matrizImagenRGB[i][j].b);
        }
        fprintf(archivoImagen, "\n");
    }

    fclose(archivoImagen);
    //----------------------------------------------------------------------------------

    FILE *archivoCoordenadas = fopen("coordenadas.txt", "w");
    if (archivoCoordenadas == NULL)
    {
        perror("Error al abrir el archivo");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numeroPixeles; i++)
    {
        fprintf(archivoCoordenadas, "(%d,%d) ", arrMensajeRGB[i].x, arrMensajeRGB[i].y);
    }

    fclose(archivoCoordenadas);
    //----------------------------------------------------------------------------------
    free(usados);
    free(arrMensajeRGB);

    for (int i = 0; i < filas; i++)
        free(matrizImagenRGB[i]);
    free(matrizImagenRGB);
}

void leerMensaje()
{
    printf("\n\t--------LECTURA--------\n");
    obtenerDimensionesXY(coordenasTXT, &numPixeles);
    obtenerDimensionesRGB(imagenTXT, &filas, &columnas);
    //----------------------------------------------------------------------------------

    RGB **matrizImagenRGB2 = leerDatosRGB(imagenTXT, filas, columnas); // Monse
    RGB *arrASCIImensaje = leerDatosXY(coordenasTXT, numPixeles);      // Monse

    for (int i = 0; i < numPixeles; i++)
    {
        int x = arrASCIImensaje[i].x, y = arrASCIImensaje[i].y;
        if ((y >= 0 && y < filas) && (x >= 0 && x < columnas))
        {
            arrASCIImensaje[i].r = matrizImagenRGB2[y][x].r;
            arrASCIImensaje[i].g = matrizImagenRGB2[y][x].g;
            arrASCIImensaje[i].b = matrizImagenRGB2[y][x].b;
            printf("\n\tEl pixel guardado en la posición %d arrASCIImensaje es: RGB(%d, %d, %d)",
                   i,
                   arrASCIImensaje[i].r,
                   arrASCIImensaje[i].g,
                   arrASCIImensaje[i].b);
        }
    }

    int c, k = 0;
    // mensaje[] ya fue declarado en la PARTE 1
    for (int i = 0; i < numPixeles; i++)
    {
        mensaje[k] = (char)arrASCIImensaje[i].r;
        mensaje[k + 1] = (char)arrASCIImensaje[i].g;
        mensaje[k + 2] = (char)arrASCIImensaje[i].b;
        k += 3;
    }
    printf("\n\n\t▶ Mensaje: ");
    for (int i = 0; i < (numPixeles * 3); i++)
    {
        printf("%c", mensaje[i]);
    }
    printf("\n");

    // liberar memoria usada en la parte de extracción
    for (int i = 0; i < filas; i++) // Monse
    {
        free(matrizImagenRGB2[i]);
    }
    free(matrizImagenRGB2);
    free(arrASCIImensaje);
}

