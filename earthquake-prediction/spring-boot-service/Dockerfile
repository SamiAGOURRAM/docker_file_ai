# Build stage
FROM maven:3.8.6-openjdk-11-slim AS build
WORKDIR /app

# Copy POM file
COPY pom.xml .

# Download all required dependencies
RUN mvn dependency:go-offline -B

# Copy source code
COPY src ./src

# Build the application
RUN mvn package -DskipTests

# Runtime stage
FROM openjdk:11-jre-slim
WORKDIR /app

# Copy the JAR file from the build stage
COPY --from=build /app/target/*.jar app.jar

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
ENTRYPOINT ["java", "-jar", "app.jar"]